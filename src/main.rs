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

    if let Some(first_line) = lines.next() { // Extract first line for filename
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
        let file = File::create(&out_path)?; // Create the output file, fails if unwritable
        let mut writer = BufWriter::with_capacity(buffer_size * 1024, file); // Buffered writer for efficiency

        // Write remaining lines with trimming, skip if too short
        for line in lines {
            if line.len() > trim {
                writer.write_all(&line[trim..])?; // Write trimmed line, fails on I/O error
                writer.write_all(b"\n")?; // Append newline, maintain format
            }
        }
    } else {
        // Handle empty chunk: use default filename if no lines
        let out_path = output_dir.join(format!("chunk_{:05}.txt", chunk.chunk_num));
        let file = File::create(&out_path)?; // Create file, fails if unwritable
        let _writer = BufWriter::with_capacity(buffer_size * 1024, file); // Buffered writer, unused here
        // No data to write; file remains empty
    }

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

    // Choose processing mode: streaming for large files (>10GB) or forced
    if args.streaming || file_size > 10 * 1024 * 1024 * 1024 {
        process_streaming(file, file_size, &output_dir, args, &dfa) // Stream for big files
    } else {
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
        .fold(
            || Vec::with_capacity(1024), // Per-thread vector, pre-allocated
            |mut acc, pos| {
                if pos == 0 || mmap[pos - 1] == b'\n' { // Check line start
                    let input = Input::new(&mmap[pos..])
                        .anchored(Anchored::Yes) // Match only at start
                        .range(..file_size - pos); // Limit search range
                    match dfa.try_search_fwd(&input) {
                        Ok(Some(_)) => acc.push(pos), // Match found, store position
                        _ => (), // No match or error, skip
                    }
                }
                acc
            },
        )
        .flatten() // Combine thread-local vectors
        .collect(); // Gather all positions

    let mut positions = positions; // Mutable positions for sorting
    positions.sort_unstable(); // Sort for sequential chunks
    positions.dedup(); // Remove duplicates from overlapping matches
    positions.push(file_size); // Append file end as last boundary

    // Process chunks in parallel, fails on first error
    positions.par_windows(2)
        .enumerate()
        .try_for_each(|(_, window)| {
            let chunk_num = counter.fetch_add(1, Ordering::Relaxed); // Assign unique number
            let chunk = Chunk {
                start: window[0], // Start of this chunk
                end: window[1], // End of this chunk
                chunk_num, // Unique identifier
            };
            process_chunk(&mmap, chunk, output_dir, args.trim, args.buffer_size) // Process it
        })?;

    println!("Processed {} chunks", positions.len() - 1); // Report chunk count
    Ok(()) // Return success or propagate errors
}

// Processes file in streaming mode, suitable for large files
fn process_streaming<A: Automaton + Send + Sync>(
    file: File, // Opened input file
    _file_size: usize, // File size (unused here, kept for consistency)
    output_dir: &PathBuf, // Directory to save chunks
    args: Args, // User-provided arguments
    dfa: &Arc<A>, // Thread-safe DFA for pattern matching
) -> Result<()> {
    let segment_size = args.segment_size * 1024 * 1024; // Segment size in bytes, controls memory
    let mut reader = BufReader::with_capacity(segment_size, file); // Buffered reader for efficiency
    let counter = Arc::new(AtomicUsize::new(0)); // Thread-safe counter for chunk numbering

    rayon::scope(|s| { // Parallel scope for processing segments
        let mut buffer = Vec::with_capacity(segment_size); // Buffer for segment data

        loop {
            let read_size = segment_size - buffer.len(); // Space left in buffer
            let mut chunk = vec![0; read_size]; // Temp buffer for reading
            match reader.read(&mut chunk) {
                Ok(0) => break, // EOF reached
                Ok(n) => buffer.extend_from_slice(&chunk[..n]), // Append read data
                Err(e) => {
                    eprintln!("Read error: {}", e); // Log error, exit loop
                    break;
                }
            }

            let mut positions = vec![]; // Store match positions in this segment
            let mut cursor = 0; // Current position in buffer
            while cursor < buffer.len() {
                if cursor == 0 || (cursor > 0 && buffer[cursor - 1] == b'\n') { // Line start check
                    let input = Input::new(&buffer[cursor..]).anchored(Anchored::Yes);
                    if let Ok(Some(_)) = dfa.try_search_fwd(&input) {
                        positions.push(cursor); // Match found, store position
                    }
                }
                cursor += 1; // Advance cursor
            }

            let chunks: Vec<_> = positions.windows(2)
                .map(|w| (w[0], w[1])) // Pair positions into chunks
                .collect();

            let output_dir = output_dir.clone(); // Clone for thread safety
            let buffer_arc: Arc<[u8]> = Arc::from(buffer.as_slice()); // Shareable buffer
            let counter_clone = counter.clone(); // Clone counter for task

            s.spawn(move |_| { // Spawn parallel task for chunk processing
                chunks.into_par_iter()
                    .for_each(|(start, end)| {
                        let chunk_num = counter_clone.fetch_add(1, Ordering::Relaxed); // Unique number
                        let chunk = Chunk {
                            start, // Chunk start
                            end, // Chunk end
                            chunk_num, // Unique identifier
                        };
                        if let Err(e) = process_chunk(
                            &*buffer_arc,
                            chunk,
                            &output_dir,
                            args.trim,
                            args.buffer_size,
                        ) {
                            eprintln!("Chunk error: {}", e); // Log processing error
                        }
                    });
            });

            // Handle buffer leftovers for next iteration
            if let Some(last) = positions.last() {
                buffer = buffer[*last..].to_vec(); // Keep unmatched tail
            } else {
                buffer.clear(); // No matches, reset buffer
            }
        }

        // Process any remaining data
        if !buffer.is_empty() {
            let output_dir = output_dir.clone(); // Clone for final task
            let buffer_arc: Arc<[u8]> = Arc::from(buffer.as_slice()); // Shareable buffer
            let counter_clone = counter.clone(); // Clone counter
            s.spawn(move |_| {
                let chunk_num = counter_clone.fetch_add(1, Ordering::Relaxed); // Final chunk number
                let chunk = Chunk {
                    start: 0, // Start at buffer beginning
                    end: buffer_arc.len(), // End at buffer end
                    chunk_num, // Unique identifier
                };
                process_chunk(&*buffer_arc, chunk, &output_dir, args.trim, args.buffer_size)
                    .unwrap_or_else(|e| eprintln!("Final chunk error: {}", e)); // Log error if fails
            });
        }
    });

    Ok(()) // Return success, errors logged inline
}
