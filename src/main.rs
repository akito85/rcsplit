use std::{
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Write},
    path::PathBuf,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

use anyhow::{Context, Result};
use clap::Parser;
use memmap2::Mmap;
use rayon::prelude::*;
use regex_automata::{
    dfa::{dense, Automaton},
    Anchored, Input,
};
use crossbeam_channel::{bounded, Sender, Receiver};

// Command-line argument structure with clap for parsing user inputs
#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "A CLI tool for splitting files based on a regex pattern.",
    long_about = "This tool splits large files into chunks based on a provided regex pattern. It supports both memory-mapped and streaming modes for efficient processing.\n\nUse the --streaming flag for large files or to enable streaming mode manually.\n\nPerformance can be tuned using buffer_size, max_memory, and segment_size options."
)]
struct Args {
    #[arg(short, long)]
    input: String, // Input file path provided by the user
    #[arg(short, long)]
    pattern: String, // Regex pattern to identify split points
    #[arg(short, long, default_value = "out_chunks")]
    output: String, // Directory where split files will be saved
    #[arg(short, long, default_value_t = 4)]
    trim: usize, // Characters to trim from the left of each line after the first
    #[arg(short, long, default_value_t = 256)]
    buffer_size: usize, // Buffer size in KB for writing, affects I/O performance
    #[arg(short, long, default_value_t = 512)]
    max_memory: usize, // Max memory in MB (currently unused, for future optimization)
    #[arg(short, long)]
    streaming: bool, // Forces streaming mode if true, else auto-decided by file size
    #[arg(short, long, default_value_t = 256)]
    segment_size: usize, // Segment size in MB for streaming mode, controls memory usage
}

// Defines a chunk of data with start/end positions and a unique number
struct Chunk {
    start: usize, // Starting byte position in the data
    end: usize, // Ending byte position in the data
    chunk_num: usize, // Unique identifier for this chunk
}

// Processes a single chunk: generates a filename from the first line, trims lines, and writes to file
fn process_chunk(
    data: &[u8], // Full data buffer (mapped or streamed) containing all bytes
    chunk: Chunk, // Chunk metadata specifying the range and number
    output_dir: &PathBuf, // Directory path for output files
    trim: usize, // Number of characters to trim from each line except the first
    buffer_size: usize, // Size of write buffer in KB, tune for I/O efficiency
) -> Result<()> {
    let chunk_data = &data[chunk.start..chunk.end]; // Slice of data for this chunk
    let mut lines = chunk_data.split(|&c| c == b'\n'); // Split chunk into lines by newline

    // Create output file based on first line
    let out_path = if let Some(first_line) = lines.next() { // Extract first line for filename
        // Convert first line to UTF-8 string, fallback to "invalid_utf8" if invalid
        let first_line_str = std::str::from_utf8(first_line).unwrap_or("invalid_utf8");
        // Sanitize filename by replacing non-alphanumeric chars with '_'
        let sanitized = first_line_str
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { '_' })
            .collect::<String>();
        // Limit filename to 100 chars and append chunk number to avoid conflicts
        let filename = if sanitized.len() > 100 {
            format!("{}_{:05}.txt", &sanitized[..100], chunk.chunk_num)
        } else {
            format!("{}_{:05}.txt", sanitized, chunk.chunk_num)
        };
        output_dir.join(filename) // Full path for the output file
    } else {
        // Handle empty chunk: use default filename if no lines
        output_dir.join(format!("chunk_{:05}.txt", chunk.chunk_num))
    };

    // Create and write to output file
    let file = File::create(&out_path)?; // Create the output file, fails if unwritable
    let mut writer = BufWriter::with_capacity(buffer_size * 1024, file); // Buffered writer for efficiency

    // Write lines with trimming, skip first line (already used for filename)
    for line in lines {
        if line.len() > trim {
            writer.write_all(&line[trim..])?; // Write trimmed line, fails on I/O error
            writer.write_all(b"\n")?; // Append newline, maintain format
        } else if !line.is_empty() {
            // Write short lines without trimming
            writer.write_all(line)?;
            writer.write_all(b"\n")?;
        }
    }

    writer.flush()?; // Ensure all data is written to disk
    Ok(()) // Return success, or propagate I/O errors
}

// Entry point: parses args, sets up dirs, and selects processing mode
fn main() -> Result<()> {
    let args = Args::parse(); // Parse CLI args, panics on invalid input
    let input_path = PathBuf::from(&args.input); // Path to input file from args
    let output_dir = PathBuf::from(&args.output); // Output directory path from args
    fs::create_dir_all(&output_dir)?; // Create output dir, fails if permissions issue

    let file = File::open(&input_path)?; // Open input file, fails if not found
    let file_size = file.metadata()?.len() as usize; // File size in bytes, fails on metadata error

    // DFA for efficient regex matching, wrapped in Arc for thread safety
    let dfa = Arc::new(
        dense::Builder::new()
            .configure(
                dense::Config::new()
                    .byte_classes(false) // Disable byte classes for speed
                    .start_kind(regex_automata::dfa::StartKind::Anchored), // Anchored for line-start matches
            )
            .build(&args.pattern) // Build DFA from pattern, fails on invalid regex
            .context("Failed to build DFA")?, // Add context to error
    );

    println!("Processing file: {} ({} bytes)", args.input, file_size);

    // Choose processing mode: streaming for large files (>10GB) or forced
    if args.streaming || file_size > 10 * 1024 * 1024 * 1024 {
        println!("Using streaming mode");
        process_streaming(file, file_size, &output_dir, args, &dfa) // Stream for big files
    } else {
        println!("Using memory-mapped mode");
        process_mapped(file, file_size, &output_dir, args, &dfa) // Memory-map for smaller files
    }
}

// Processes file using memory mapping, ideal for smaller files
fn process_mapped<A: Automaton + Send + Sync>(
    file: File, // Opened input file
    file_size: usize, // Total size of the file in bytes
    output_dir: &PathBuf, // Directory to save chunks
    args: Args, // User-provided arguments
    dfa: &Arc<A>, // Thread-safe DFA for pattern matching
) -> Result<()> {
    let mmap = unsafe { Mmap::map(&file)? }; // Memory-map file, fails on mapping error
    let counter = Arc::new(AtomicUsize::new(0)); // Thread-safe counter for chunk numbering

    // Find pattern matches at line starts in parallel
    let positions: Vec<usize> = (0..file_size)
        .into_par_iter() // Parallel iteration over file bytes
        .filter(|&pos| {
            // Check if position is 0 or previous byte is a newline
            pos == 0 || (pos > 0 && mmap[pos - 1] == b'\n')
        })
        .filter_map(|pos| {
            let input = Input::new(&mmap[pos..])
                .anchored(Anchored::Yes) // Match only at start
                .range(..file_size.saturating_sub(pos)); // Limit search range
            match dfa.try_search_fwd(&input) {
                Ok(Some(_)) => Some(pos), // Match found, return position
                _ => None, // No match or error, skip
            }
        })
        .collect(); // Gather all positions

    // Start with position 0 if not included
    let mut positions = if !positions.contains(&0) {
        let mut pos = positions;
        pos.insert(0, 0);
        pos
    } else {
        positions
    };

    positions.sort_unstable(); // Sort for sequential chunks
    positions.dedup(); // Remove duplicates from overlapping matches

    // Ensure file end is included
    if !positions.contains(&file_size) {
        positions.push(file_size);
    }

    println!("Found {} split points", positions.len() - 1);

    // Generate chunks from position pairs and process in parallel
    positions.windows(2)
        .enumerate()
        .par_bridge() // Convert to parallel iterator
        .try_for_each(|(_, window)| {
            let chunk_num = counter.fetch_add(1, Ordering::Relaxed); // Assign unique number
            let chunk = Chunk {
                start: window[0], // Start of this chunk
                end: window[1], // End of this chunk
                chunk_num, // Unique identifier
            };
            process_chunk(&mmap, chunk, output_dir, args.trim, args.buffer_size) // Process it
        })?;

    println!("Processed {} chunks", counter.load(Ordering::Relaxed)); // Report chunk count
    Ok(()) // Return success or propagate errors
}

// Processes file in streaming mode, suitable for large files
fn process_streaming<A: Automaton + Send + Sync>(
    file: File, // Opened input file
    file_size: usize, // File size in bytes
    output_dir: &PathBuf, // Directory to save chunks
    args: Args, // User-provided arguments
    dfa: &Arc<A>, // Thread-safe DFA for pattern matching
) -> Result<()> {
    let segment_size = args.segment_size * 1024 * 1024; // Segment size in bytes
    let mut reader = BufReader::with_capacity(segment_size, file); // Buffered reader
    let counter = Arc::new(AtomicUsize::new(0)); // Thread-safe counter for chunk numbering

    // Create channels for chunk processing coordination
    let (sender, receiver): (Sender<(Vec<u8>, Vec<usize>)>, Receiver<(Vec<u8>, Vec<usize>)>) =
        bounded(4); // Limit channel capacity to prevent memory overflow

    // Spawn worker threads for chunk processing
    let worker_count = rayon::current_num_threads().min(4); // Limit worker threads
    let workers = (0..worker_count).map(|_| {
        let rx = receiver.clone();
        let out_dir = output_dir.clone();
        let counter_clone = counter.clone();
        let args_trim = args.trim;
        let args_buffer_size = args.buffer_size;

        std::thread::spawn(move || {
            while let Ok((buffer, positions)) = rx.recv() {
                // Process each chunk in this segment
                for window in positions.windows(2) {
                    let chunk_num = counter_clone.fetch_add(1, Ordering::Relaxed);
                    let chunk = Chunk {
                        start: window[0],
                        end: window[1],
                        chunk_num,
                    };

                    if let Err(e) = process_chunk(
                        &buffer,
                        chunk,
                        &out_dir,
                        args_trim,
                        args_buffer_size,
                    ) {
                        eprintln!("Chunk error: {}", e);
                    }
                }
            }
        })
    }).collect::<Vec<_>>();

    // Buffer for accumulated data across segments
    let mut buffer = Vec::with_capacity(segment_size * 2);
    let mut total_bytes_read = 0;
    let mut last_match_pos = 0;

    // Main processing loop
    loop {
        // Make room in buffer if needed
        if buffer.len() >= segment_size {
            // Shift buffer to remove processed chunks
            if last_match_pos > 0 {
                buffer.drain(0..last_match_pos);
                last_match_pos = 0;
            }
        }

        // Read next chunk from file
        let mut chunk = vec![0; segment_size];
        let bytes_read = match reader.read(&mut chunk) {
            Ok(0) => break, // EOF reached
            Ok(n) => {
                total_bytes_read += n;
                println!("Read {} bytes ({:.2}% of file)",
                    total_bytes_read,
                    (total_bytes_read as f64 / file_size as f64) * 100.0);
                chunk.truncate(n);
                n
            }
            Err(e) => {
                eprintln!("Read error: {}", e);
                break;
            }
        };

        // Append to buffer
        buffer.extend_from_slice(&chunk);

        // Find matches in full buffer
        let mut positions = vec![];
        for i in 0..buffer.len() {
            if i == 0 || buffer[i - 1] == b'\n' { // Line start
                let input = Input::new(&buffer[i..])
                    .anchored(Anchored::Yes);
                if let Ok(Some(_)) = dfa.try_search_fwd(&input) {
                    positions.push(i);
                }
            }
        }

        // Process if we have matches
        if !positions.is_empty() {
            // Add 0 if first match isn't at start
            if positions[0] > 0 {
                positions.insert(0, 0);
            }

            // Clone for sending
            let last_pos = *positions.last().unwrap_or(&0);
            let send_buffer = buffer[0..last_pos].to_vec();

            // Send buffer and positions to workers
            if sender.send((send_buffer, positions.clone())).is_err() {
                eprintln!("Worker channel closed unexpectedly");
                break;
            }

            // Update last match position for buffer management
            last_match_pos = last_pos;
        }

        // Check if we're at EOF
        if bytes_read < segment_size {
            break;
        }
    }

    // Process any remaining data
    if !buffer.is_empty() {
        let mut positions = vec![0];
        positions.push(buffer.len());

        if sender.send((buffer, positions)).is_err() {
            eprintln!("Failed to send final chunk");
        }
    }

    // Drop sender to signal workers to exit
    drop(sender);

    // Wait for workers to finish
    for worker in workers {
        if let Err(e) = worker.join() {
            eprintln!("Worker thread panicked: {:?}", e);
        }
    }

    println!("Processed {} chunks", counter.load(Ordering::Relaxed));
    Ok(())
}
