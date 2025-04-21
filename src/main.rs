use std::{
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Write},
    path::PathBuf,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use anyhow::{Context, Result};
use clap::Parser;
use crossbeam_channel::bounded;
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
    input: String, // Input file path
    #[arg(short, long)]
    pattern: String, // Regex pattern for split points
    #[arg(short, long, default_value = "out_chunks")]
    output: String, // Output directory for chunks
    #[arg(short, long, default_value_t = 4)]
    trim: usize, // Characters to trim from subsequent lines
    #[arg(short, long, default_value_t = 256)]
    buffer_size: usize, // Buffer size in KB for I/O
    #[arg(short, long, default_value_t = 512)]
    max_memory: usize, // Max memory in MB for streaming
    #[arg(short, long)]
    streaming: bool, // Force streaming mode
    #[arg(short, long, default_value_t = 256)]
    segment_size: usize, // Segment size in MB for streaming
    #[arg(short, long, default_value_t = false)]
    verbose: bool, // Enable verbose logging
    #[arg(long, default_value_t = 0)]
    threads: usize, // Number of threads (0 = auto)
    #[arg(long, default_value_t = 300)]
    timeout_secs: u64, // Timeout for worker threads in seconds
}

// Defines a chunk of data with start/end positions and unique number
#[derive(Debug, Clone)]
struct Chunk {
    start: usize, // Start byte position
    end: usize, // End byte position
    chunk_num: usize, // Unique chunk identifier
}

// Structure for shared buffer in streaming mode
struct SharedBuffer {
    data: Arc<Vec<u8>>, // Shared data buffer
    positions: Vec<usize>, // Split positions
    segment_offset: usize, // Offset in file
}

// Generate a sanitized filename from chunk content
fn generate_filename(data: &[u8], chunk_num: usize) -> String {
    let mut lines = Vec::with_capacity(2);
    let mut start = 0;
    let mut line_count = 0;

    while start < data.len() && line_count < 2 {
        let line_end = data[start..]
            .iter()
            .position(|&b| b == b'\n')
            .map(|p| p + start)
            .unwrap_or(data.len());

        lines.push(&data[start..line_end]);
        line_count += 1;
        start = line_end + 1;

        if start >= data.len() {
            break;
        }
    }

    let name_line = if !lines.is_empty() && lines[0].first() == Some(&b'*') {
        lines[0]
    } else if lines.len() > 1 && lines[1].first() == Some(&b'*') {
        lines[1]
    } else if !lines.is_empty() {
        lines[0]
    } else {
        "empty".as_bytes()
    };

    let line_str = std::str::from_utf8(name_line).unwrap_or("invalid_utf8");
    let sanitized = line_str
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '_' })
        .collect::<String>();

    let prefix = if sanitized.len() > 100 {
        &sanitized[..100]
    } else {
        &sanitized
    };

    format!("{}_{:05}.txt", prefix, chunk_num)
}

// Process a chunk: generate filename, write all lines, trim subsequent lines
fn process_chunk(
    data: &[u8],
    chunk: Chunk,
    output_dir: &PathBuf,
    trim: usize,
    buffer_size: usize,
    verbose: bool,
) -> Result<()> {
    if chunk.start >= data.len() || chunk.end > data.len() || chunk.start >= chunk.end {
        if verbose {
            eprintln!("Skipping invalid chunk: {:?}", chunk);
        }
        return Ok(());
    }

    let chunk_data = &data[chunk.start..chunk.end];
    let filename = generate_filename(chunk_data, chunk.chunk_num);
    let out_path = output_dir.join(&filename);

    let file = File::create(&out_path)
        .with_context(|| format!("Failed to create file: {}", out_path.display()))?;
    let mut writer = BufWriter::with_capacity(buffer_size * 1024, file);

    // Split chunk into lines (excluding newlines)
    let lines: Vec<&[u8]> = chunk_data.split(|&b| b == b'\n').collect();

    // Process each line
    for (i, line) in lines.iter().enumerate() {
        if line.is_empty() && i == lines.len() - 1 {
            // Skip trailing empty line
            continue;
        }

        // Write first line fully, trim subsequent lines
        if i == 0 {
            writer.write_all(line)?;
        } else if line.len() > trim {
            writer.write_all(&line[trim..])?;
        } else if !line.is_empty() {
            writer.write_all(line)?;
        }

        // Add newline unless it's the last line
        if i < lines.len() - 1 {
            writer.write_all(b"\n")?;
        }
    }

    writer.flush()?;
    if verbose {
        println!("Wrote chunk {} to {}", chunk.chunk_num, out_path.display());
    }

    Ok(())
}

// Entry point: parse args, set up, and select processing mode
fn main() -> Result<()> {
    let args = Args::parse();

    if args.threads > 1024 {
        eprintln!("Warning: High thread count ({}). May impact performance.", args.threads);
    }
    if args.buffer_size > 10240 {
        eprintln!("Warning: Large buffer size ({}KB). May increase memory usage.", args.buffer_size);
    }

    // Set up thread pool
    let thread_count = if args.threads > 0 {
        args.threads.min(num_cpus::get().max(1))
    } else {
        num_cpus::get().max(1)
    };
    rayon::ThreadPoolBuilder::new()
        .num_threads(thread_count)
        .build_global()
        .context("Failed to initialize thread pool")?;

    let input_path = PathBuf::from(&args.input);
    if !input_path.exists() {
        return Err(anyhow::anyhow!("Input file does not exist: {}", input_path.display()));
    }

    let output_dir = PathBuf::from(&args.output);
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("Failed to create directory: {}", output_dir.display()))?;

    let file = File::open(&input_path)
        .with_context(|| format!("Failed to open file: {}", input_path.display()))?;
    let file_size = file.metadata()?.len() as usize;

    let split_dfa = Arc::new(
        dense::Builder::new()
            .configure(
                dense::Config::new()
                    .byte_classes(true)
                    .start_kind(regex_automata::dfa::StartKind::Anchored),
            )
            .build(&args.pattern)
            .context("Failed to build regex pattern")?,
    );

    if args.verbose {
        println!("Processing file: {} ({} bytes)", args.input, file_size);
    }

    if file_size == 0 {
        println!("Empty input file. Nothing to process.");
        return Ok(());
    }

    let max_memory_bytes = args.max_memory * 1024 * 1024;
    if args.streaming || file_size > max_memory_bytes {
        if args.verbose {
            println!("Using streaming mode");
        }
        process_streaming(file, file_size, &output_dir, &args, &split_dfa)
    } else {
        if args.verbose {
            println!("Using memory-mapped mode");
        }
        process_mapped(file, file_size, &output_dir, &args, &split_dfa)
    }
}

// Process file using memory mapping
fn process_mapped<A: Automaton + Send + Sync>(
    file: File,
    file_size: usize,
    output_dir: &PathBuf,
    args: &Args,
    split_dfa: &Arc<A>,
) -> Result<()> {
    let mmap = unsafe { Mmap::map(&file)? };
    let counter = AtomicUsize::new(0);

    let positions = find_split_positions(&mmap, file_size, split_dfa, args.verbose)?;
    process_chunks(&mmap, positions, output_dir, &counter, args)
}

// Find split positions in the data
fn find_split_positions<A: Automaton + Send + Sync>(
    data: &[u8],
    file_size: usize,
    dfa: &Arc<A>,
    verbose: bool,
) -> Result<Vec<usize>> {
    if file_size == 0 {
        return Ok(vec![0]);
    }

    let chunk_size = (file_size / rayon::current_num_threads().max(1)).max(1024 * 1024);
    let chunk_count = (file_size + chunk_size - 1) / chunk_size;

    if verbose {
        println!("Searching for matches in {} chunks", chunk_count);
    }

    let positions: Vec<usize> = (0..chunk_count)
        .into_par_iter()
        .flat_map(|i| {
            let start = i * chunk_size;
            let end = (start + chunk_size).min(file_size);
            let mut matches = Vec::with_capacity(1024);

            if i == 0 {
                matches.push(0);
            }

            for pos in start..end {
                if (pos == 0 || data.get(pos - 1) == Some(&b'\n')) && pos < file_size {
                    let input = Input::new(&data[pos..])
                        .anchored(Anchored::Yes)
                        .range(..file_size - pos);
                    if matches!(dfa.try_search_fwd(&input), Ok(Some(_))) {
                        matches.push(pos);
                    }
                }
            }

            matches
        })
        .collect();

    let mut result = positions;
    result.sort_unstable();
    result.dedup();
    if !result.contains(&file_size) {
        result.push(file_size);
    }

    if verbose {
        println!("Found {} split points", result.len() - 1);
    }

    Ok(result)
}

// Process chunks in parallel
fn process_chunks(
    data: &[u8],
    positions: Vec<usize>,
    output_dir: &PathBuf,
    counter: &AtomicUsize,
    args: &Args,
) -> Result<()> {
    let chunks: Vec<_> = positions
        .windows(2)
        .map(|window| Chunk {
            start: window[0],
            end: window[1],
            chunk_num: counter.fetch_add(1, Ordering::Relaxed),
        })
        .collect();

    chunks.into_par_iter().try_for_each(|chunk| {
        process_chunk(
            data,
            chunk,
            output_dir,
            args.trim,
            args.buffer_size,
            args.verbose,
        )
    })?;

    if args.verbose {
        println!("Processed {} chunks", counter.load(Ordering::Relaxed));
    }

    Ok(())
}

// Process file in streaming mode
fn process_streaming<A: Automaton + Send + Sync + 'static>(
    file: File,
    file_size: usize,
    output_dir: &PathBuf,
    args: &Args,
    split_dfa: &Arc<A>,
) -> Result<()> {
    let segment_size = args.segment_size * 1024 * 1024;
    let processed_bytes = Arc::new(AtomicUsize::new(0));
    let channel_size = (args.max_memory * 1024 * 1024 / segment_size)
        .max(2)
        .min(16);
    let (chunk_sender, chunk_receiver) = bounded::<SharedBuffer>(channel_size);
    let (done_sender, done_receiver) = bounded::<()>(1);
    let counter = Arc::new(AtomicUsize::new(0));

    let worker_count = rayon::current_num_threads().min(8).max(1);
    let _workers: Vec<_> = (0..worker_count)
        .map(|worker_id| {
            let rx = chunk_receiver.clone();
            let output_dir = output_dir.clone();
            let counter = counter.clone();
            let processed_bytes = processed_bytes.clone();
            let file_size = file_size;
            let verbose = args.verbose;
            let trim = args.trim;
            let buffer_size = args.buffer_size;
            let done_tx = done_sender.clone();
            let timeout = Duration::from_secs(args.timeout_secs);

            std::thread::spawn(move || -> Result<()> {
                loop {
                    match rx.recv_timeout(timeout) {
                        Ok(shared_buffer) => {
                            let positions = &shared_buffer.positions;
                            let segment_offset = shared_buffer.segment_offset;

                            if positions.len() < 2 {
                                continue;
                            }

                            if verbose && worker_id == 0 {
                                let bytes = processed_bytes.load(Ordering::Relaxed);
                                let progress = (bytes as f64 / file_size as f64) * 100.0;
                                println!(
                                    "Progress: {:.2}% ({}/{} bytes)",
                                    progress, bytes, file_size
                                );
                            }

                            for window in positions.windows(2) {
                                let chunk = Chunk {
                                    start: window[0] - segment_offset,
                                    end: window[1] - segment_offset,
                                    chunk_num: counter.fetch_add(1, Ordering::Relaxed),
                                };

                                if chunk.start >= shared_buffer.data.len()
                                    || chunk.end > shared_buffer.data.len()
                                    || chunk.start >= chunk.end
                                {
                                    if verbose {
                                        eprintln!(
                                            "Invalid chunk bounds: start={}, end={}, buffer_len={}",
                                            chunk.start, chunk.end, shared_buffer.data.len()
                                        );
                                    }
                                    continue;
                                }

                                if let Err(e) = process_chunk(
                                    shared_buffer.data.as_slice(),
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
                        Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                            if rx.is_empty() {
                                break;
                            }
                            if verbose && worker_id == 0 {
                                println!("Worker {} timeout, continuing...", worker_id);
                            }
                        }
                        Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
                    }
                }

                let _ = done_tx.send(());
                Ok(())
            })
        })
        .collect();

    let reader_handle = {
        let split_dfa = split_dfa.clone();
        let sender = chunk_sender.clone();
        let processed_bytes = processed_bytes.clone();
        let verbose = args.verbose;

        std::thread::spawn(move || -> Result<()> {
            let mut reader = BufReader::with_capacity(segment_size, file);
            let mut absolute_pos = 0;
            let mut at_line_start = true;

            loop {
                let mut buffer = vec![0u8; segment_size];
                let bytes_read = reader.read(&mut buffer)?;
                if bytes_read == 0 {
                    break;
                }
                buffer.truncate(bytes_read);

                let positions =
                    find_split_positions_in_segment(&buffer, &*split_dfa, at_line_start, absolute_pos);

                // Calculate next at_line_start before moving buffer
                let next_at_line_start = buffer.last().map_or(false, |&b| b == b'\n');

                let shared_buffer = SharedBuffer {
                    data: Arc::new(buffer),
                    positions,
                    segment_offset: absolute_pos,
                };

                sender.send(shared_buffer)?;
                processed_bytes.fetch_add(bytes_read, Ordering::Relaxed);
                at_line_start = next_at_line_start;
                absolute_pos += bytes_read;
            }

            Ok(())
        })
    };

    drop(chunk_sender);
    reader_handle.join().unwrap()?;
    for _ in 0..worker_count {
        done_receiver.recv().ok();
    }

    Ok(())
}

// Find split positions within a segment
fn find_split_positions_in_segment<A: Automaton>(
    data: &[u8],
    dfa: &A,
    at_line_start: bool,
    segment_offset: usize,
) -> Vec<usize> {
    let mut positions = Vec::with_capacity(1024);

    let mut potential_starts = Vec::with_capacity(data.len() / 128);
    if at_line_start {
        potential_starts.push(0);
    }
    for k in 1..data.len() {
        if data[k - 1] == b'\n' {
            potential_starts.push(k);
        }
    }

    for &j in &potential_starts {
        let input = Input::new(&data[j..]).anchored(Anchored::Yes);
        if matches!(dfa.try_search_fwd(&input), Ok(Some(_))) {
            positions.push(segment_offset + j);
        }
    }

    positions
}
