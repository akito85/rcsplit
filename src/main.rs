use std::{
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Write},
    path::PathBuf,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
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
    max_memory: usize, // Max memory in MB (limits memory usage in streaming mode)
    #[arg(short, long)]
    streaming: bool, // Forces streaming mode if true, else auto-decided by file size
    #[arg(short, long, default_value_t = 256)]
    segment_size: usize, // Segment size in MB for streaming mode, controls memory usage
    #[arg(short, long, default_value_t = false)]
    verbose: bool, // Enable verbose logging
    #[arg(long, default_value_t = 0)]
    threads: usize, // Number of threads (0 = auto)
}

// Defines a chunk of data with start/end positions and a unique number
#[derive(Debug, Clone)]
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
    verbose: bool, // Enable verbose logging
) -> Result<()> {
    // Quick bounds check to avoid panics
    if chunk.start >= data.len() || chunk.end > data.len() || chunk.start >= chunk.end {
        if verbose {
            eprintln!("Skipping invalid chunk: {:?}, data len: {}", chunk, data.len());
        }
        return Ok(());
    }

    let chunk_data = &data[chunk.start..chunk.end]; // Slice of data for this chunk

    // Find the first newline to get the first line
    let first_line_end = chunk_data.iter()
        .position(|&b| b == b'\n')
        .unwrap_or(chunk_data.len());

    let first_line = &chunk_data[..first_line_end];

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

    let out_path = output_dir.join(filename); // Full path for the output file

    // Create and write to output file
    let file = File::create(&out_path)?; // Create the output file, fails if unwritable
    let mut writer = BufWriter::with_capacity(buffer_size * 1024, file); // Buffered writer for efficiency

    // If there's content after the first line
    if first_line_end < chunk_data.len() {
        // Start after the first line's newline
        let mut pos = first_line_end + 1;

        while pos < chunk_data.len() {
            // Find next newline
            let line_end = chunk_data[pos..]
                .iter()
                .position(|&b| b == b'\n')
                .map(|p| p + pos)
                .unwrap_or(chunk_data.len());

            let line = &chunk_data[pos..line_end];

            // Apply trimming if line is long enough
            if line.len() > trim {
                writer.write_all(&line[trim..])?;
            } else if !line.is_empty() {
                writer.write_all(line)?;
            }

            // Add newline if not the last line
            if line_end < chunk_data.len() {
                writer.write_all(b"\n")?;
            }

            // Move to position after this newline
            pos = line_end + 1;

            // Break if we're at the end
            if pos >= chunk_data.len() {
                break;
            }
        }
    }

    writer.flush()?; // Ensure all data is written to disk

    if verbose {
        println!("Wrote chunk {} to {}", chunk.chunk_num, out_path.display());
    }

    Ok(())
}

// Entry point: parses args, sets up dirs, and selects processing mode
fn main() -> Result<()> {
    let args = Args::parse(); // Parse CLI args, panics on invalid input

    // Set up thread pool with explicit thread count if specified
    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .context("Failed to initialize thread pool")?;
    }

    let input_path = PathBuf::from(&args.input); // Path to input file from args
    let output_dir = PathBuf::from(&args.output); // Output directory path from args
    fs::create_dir_all(&output_dir)?; // Create output dir, fails if permissions issue

    let file = File::open(&input_path)?; // Open input file, fails if not found
    let file_size = file.metadata()?.len() as usize; // File size in bytes, fails on metadata error

    // Pre-compile regex DFA once, share across threads
    let dfa = Arc::new(
        dense::Builder::new()
            .configure(
                dense::Config::new()
                    .byte_classes(true) // Enable byte classes for faster matching
                    .start_kind(regex_automata::dfa::StartKind::Anchored) // Anchored for line-start matches
                    .prefilter(true) // Enable prefiltering for faster matching
            )
            .build(&args.pattern)
            .context("Failed to build DFA")?
    );

    if args.verbose {
        println!("Processing file: {} ({} bytes)", args.input, file_size);
    }

    // Choose processing mode: streaming for large files (>available memory) or forced
    let max_memory_bytes = args.max_memory * 1024 * 1024;
    if args.streaming || file_size > max_memory_bytes {
        if args.verbose {
            println!("Using streaming mode");
        }
        process_streaming(file, file_size, &output_dir, &args, &dfa) // Stream for big files
    } else {
        if args.verbose {
            println!("Using memory-mapped mode");
        }
        process_mapped(file, file_size, &output_dir, &args, &dfa) // Memory-map for smaller files
    }
}

// Processes file using memory mapping, ideal for smaller files
fn process_mapped<A: Automaton + Send + Sync>(
    file: File, // Opened input file
    file_size: usize, // Total size of the file in bytes
    output_dir: &PathBuf, // Directory to save chunks
    args: &Args, // User-provided arguments
    dfa: &Arc<A>, // Thread-safe DFA for pattern matching
) -> Result<()> {
    let mmap = unsafe { Mmap::map(&file)? }; // Memory-map file, fails on mapping error
    let counter = AtomicUsize::new(0); // Thread-safe counter for chunk numbering

    // Pre-calculate split points in parallel
    // This step leverages multi-core processing
    let positions = find_split_positions(&mmap, file_size, dfa, args.verbose)?;

    // Process chunks in parallel
    process_chunks(&mmap, positions, output_dir, &counter, args)
}

// Find all positions where the pattern matches at the start of a line
fn find_split_positions<A: Automaton + Send + Sync>(
    data: &[u8],
    file_size: usize,
    dfa: &Arc<A>,
    verbose: bool,
) -> Result<Vec<usize>> {
    // Calculate optimal chunk size for parallelism
    // Use smaller chunks for better load balancing
    let chunk_size = (file_size / rayon::current_num_threads()).max(1024 * 1024);
    let chunk_count = (file_size + chunk_size - 1) / chunk_size;

    if verbose {
        println!("Searching for pattern matches in {} chunks", chunk_count);
    }

    // Process file in chunks to improve cache locality
    let positions: Vec<usize> = (0..chunk_count)
        .into_par_iter()
        .flat_map(|i| {
            let start = i * chunk_size;
            let end = (start + chunk_size).min(file_size);
            let mut matches = Vec::with_capacity(1024);

            // Always include the first position in the first chunk
            if i == 0 {
                matches.push(0);
            }

            // Find all matches in this chunk
            for pos in start..end {
                // Check if this is a line start (first byte or follows newline)
                if (pos == 0 || data[pos-1] == b'\n') && pos < file_size {
                    let input = Input::new(&data[pos..])
                        .anchored(Anchored::Yes)
                        .range(..file_size - pos);

                    // Check for pattern match
                    if dfa.try_search_fwd(&input).map_or(false, |o| o.is_some()) {
                        matches.push(pos);
                    }
                }
            }

            matches
        })
        .collect();

    // Create an ordered, deduplicated list of positions
    let mut result = positions;
    result.sort_unstable();
    result.dedup();

    // Always include the file end
    if !result.contains(&file_size) {
        result.push(file_size);
    }

    if verbose {
        println!("Found {} split points", result.len() - 1);
    }

    Ok(result)
}

// Process chunks in parallel based on split positions
fn process_chunks(
    data: &[u8],
    positions: Vec<usize>,
    output_dir: &PathBuf,
    counter: &AtomicUsize,
    args: &Args,
) -> Result<()> {
    // Create chunks from adjacent positions
    let chunks: Vec<_> = positions.windows(2)
        .map(|window| {
            Chunk {
                start: window[0],
                end: window[1],
                chunk_num: counter.fetch_add(1, Ordering::Relaxed),
            }
        })
        .collect();

    // Process chunks in parallel
    chunks.into_par_iter()
        .try_for_each(|chunk| {
            process_chunk(
                data,
                chunk,
                output_dir,
                args.trim,
                args.buffer_size,
                args.verbose
            )
        })?;

    if args.verbose {
        println!("Processed {} chunks", counter.load(Ordering::Relaxed));
    }

    Ok(())
}

// Processes file in streaming mode, suitable for large files
fn process_streaming<A: Automaton + Send + Sync>(
    file: File, // Opened input file
    file_size: usize, // File size in bytes
    output_dir: &PathBuf, // Directory to save chunks
    args: &Args, // User-provided arguments
    dfa: &Arc<A>, // Thread-safe DFA for pattern matching
) -> Result<()> {
    // Calculate optimal segment size based on available memory
    let segment_size = args.segment_size * 1024 * 1024;

    // Track bytes processed for progress reporting
    let processed_bytes = Arc::new(AtomicUsize::new(0));

    // Create bounded channel for work distribution
    // Use channel size based on available memory and thread count
    let channel_size = (args.max_memory * 1024 * 1024 / segment_size)
        .max(2)  // At least 2 slots
        .min(16); // No more than 16 slots to limit memory usage

    let (chunk_sender, chunk_receiver) = bounded::<(Vec<u8>, Vec<usize>)>(channel_size);

    // Create progress counter
    let counter = Arc::new(AtomicUsize::new(0));

    // Start worker threads pool
    let worker_count = rayon::current_num_threads().min(8); // Limit worker count
    let workers: Vec<_> = (0..worker_count)
        .map(|worker_id| {
            let rx = chunk_receiver.clone();
            let output_dir = output_dir.clone();
            let counter = counter.clone();
            let processed_bytes = processed_bytes.clone();
            let file_size = file_size;
            let verbose = args.verbose;
            let trim = args.trim;
            let buffer_size = args.buffer_size;

            std::thread::spawn(move || -> Result<()> {
                // Process chunks as they arrive
                while let Ok((buffer, positions)) = rx.recv() {
                    if positions.len() < 2 {
                        continue; // Need at least start and end positions
                    }

                    // Track progress
                    if verbose && worker_id == 0 {
                        let bytes = processed_bytes.load(Ordering::Relaxed);
                        let progress = (bytes as f64 / file_size as f64) * 100.0;
                        println!("Progress: {:.2}% ({}/{} bytes)",
                                progress, bytes, file_size);
                    }

                    // Process each chunk in this segment
                    for window in positions.windows(2) {
                        let chunk = Chunk {
                            start: window[0],
                            end: window[1],
                            chunk_num: counter.fetch_add(1, Ordering::Relaxed),
                        };

                        if let Err(e) = process_chunk(
                            &buffer,
                            chunk,
                            &output_dir,
                            trim,
                            buffer_size,
                            verbose,
                        ) {
                            eprintln!("Chunk error: {}", e);
                        }
                    }
                }

                Ok(())
            })
        })
        .collect();

    // Producer thread: read file and find split points
    let reader_thread = {
        let dfa = dfa.clone();
        let sender = chunk_sender.clone();
        let processed_bytes = processed_bytes.clone();
        let verbose = args.verbose;

        std::thread::spawn(move || -> Result<()> {
            // Use a large buffer for efficient reading
            let mut reader = BufReader::with_capacity(segment_size, file);

            // Main state variables
            let mut buffer = Vec::with_capacity(segment_size * 2);
            let mut last_match_pos = 0;
            let mut eof_reached = false;

            while !eof_reached {
                // Handle buffer management
                if last_match_pos > 0 && buffer.len() >= segment_size {
                    // Shift buffer to remove processed data
                    buffer.drain(0..last_match_pos);
                    last_match_pos = 0;
                }

                // Read next segment
                let bytes_to_read = segment_size - (buffer.len() - last_match_pos);
                let mut chunk = vec![0; bytes_to_read];

                match reader.read(&mut chunk) {
                    Ok(0) => {
                        eof_reached = true;
                    },
                    Ok(n) => {
                        chunk.truncate(n);
                        buffer.extend_from_slice(&chunk);
                        processed_bytes.fetch_add(n, Ordering::Relaxed);
                    },
                    Err(e) => {
                        eprintln!("Read error: {}", e);
                        break;
                    }
                }

                // Find pattern matches in buffer
                let mut positions = Vec::with_capacity(1024);
                positions.push(0); // Always include start position

                for i in last_match_pos..buffer.len() {
                    if i == 0 || buffer[i - 1] == b'\n' {
                        let input = Input::new(&buffer[i..])
                            .anchored(Anchored::Yes);

                        if dfa.try_search_fwd(&input).map_or(false, |o| o.is_some()) {
                            positions.push(i);
                        }
                    }
                }

                // If at EOF, add buffer end as final position
                if eof_reached {
                    positions.push(buffer.len());
                }

                // Sort and deduplicate positions
                positions.sort_unstable();
                positions.dedup();

                // If there are multiple positions, send buffer for processing
                if positions.len() > 1 {
                    if verbose {
                        println!("Found {} split points in current segment", positions.len() - 1);
                    }

                    // Skip processing if last position equals first position
                    if positions.len() >= 2 && positions[0] != positions[positions.len() - 1] {
                        let data_to_send = buffer.clone();

                        if let Err(e) = sender.send((data_to_send, positions.clone())) {
                            eprintln!("Failed to send chunk: {}", e);
                            break;
                        }

                        // Update last match position
                        if let Some(&pos) = positions.last() {
                            last_match_pos = pos;
                        }
                    }
                }

                // Break if EOF and buffer processed
                if eof_reached {
                    break;
                }
            }

            // Process any remaining data if not empty
            if !buffer.is_empty() && last_match_pos < buffer.len() {
                let mut positions = vec![last_match_pos, buffer.len()];
                sender.send((buffer, positions))?;
            }

            Ok(())
        })
    };

    // Wait for reader to finish, then drop sender to signal workers
    if let Err(e) = reader_thread.join() {
        eprintln!("Reader thread panicked: {:?}", e);
    }

    // Signal end of work by dropping the sender
    drop(chunk_sender);

    // Wait for workers to finish
    for (i, worker) in workers.into_iter().enumerate() {
        if let Err(e) = worker.join() {
            eprintln!("Worker {} thread panicked: {:?}", i, e);
        }
    }

    if args.verbose {
        println!("Processed {} chunks", counter.load(Ordering::Relaxed));
    }

    Ok(())
}
