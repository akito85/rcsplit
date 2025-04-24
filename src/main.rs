use std::{
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Write, Seek},
    path::PathBuf,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

use anyhow::{Context, Result};
use clap::Parser;
use crossbeam_channel::bounded;
use indicatif::{ProgressBar, ProgressStyle};
use memmap2::Mmap;
use rayon::prelude::*;
use regex_automata::{
    dfa::{dense, Automaton},
    Anchored, Input,
};
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "A CLI tool for splitting files based on a regex pattern.",
    long_about = "Splits large files into chunks based on a regex pattern. Supports memory-mapped and streaming modes.\n\nUse --streaming for large files or to force streaming mode.\n\nTune performance with buffer_size, max_memory, and segment_size."
)]
struct Args {
    #[arg(short, long)]
    input: String,
    #[arg(short, long)]
    pattern: String,
    #[arg(short, long, default_value = "out_chunks")]
    output: String,
    #[arg(short, long, default_value_t = 4)]
    trim: usize,
    #[arg(short, long, default_value_t = 256)]
    buffer_size: usize,
    #[arg(short, long, default_value_t = 512)]
    max_memory: usize,
    #[arg(short, long)]
    streaming: bool,
    #[arg(short, long, default_value_t = 256)]
    segment_size: usize,
    #[arg(short, long, default_value_t = false)]
    verbose: bool,
    #[arg(long, default_value_t = 0)]
    threads: usize,
    #[arg(long, default_value_t = 300)]
    timeout_secs: u64,
}

#[derive(Debug, Clone)]
struct Chunk {
    start: usize,
    end: usize,
    chunk_num: usize,
}

struct SharedBuffer {
    data: Arc<Vec<u8>>,
    positions: Vec<usize>,
    segment_offset: usize,
}

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

    let name_line = if !lines.is_empty() && lines[0].starts_with(b"*") {
        lines[0]
    } else if lines.len() > 1 && lines[1].starts_with(b"*") {
        lines[1]
    } else if !lines.is_empty() {
        lines[0]
    } else {
        b"empty"
    };

    let name_str = std::str::from_utf8(name_line).unwrap_or("invalid_utf8");
    let base_name = if name_str.starts_with('*') {
        &name_str[1..]
    } else {
        name_str
    }.trim();

    let sanitized = base_name.chars().map(|c| if c.is_alphanumeric() { c } else { '_' }).collect::<String>();
    let without_leading = sanitized.trim_start_matches('_');
    let final_base = if without_leading.is_empty() {
        "chunk".to_string()
    } else {
        without_leading.chars().take(100).collect()
    };

    format!("{}_{:05}.000", final_base, chunk_num)
}

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

    let lines: Vec<&[u8]> = chunk_data.split(|&b| b == b'\n').collect();
    if lines.len() > 2 {
        let content_lines = &lines[2..];
        for (i, line) in content_lines.iter().enumerate() {
            if line.len() > trim {
                writer.write_all(&line[trim..])?;
            } else {
                writer.write_all(line)?;
            }
            if i < content_lines.len() - 1 {
                writer.write_all(b"\n")?;
            }
        }
    }

    writer.flush()?;
    if verbose {
        println!("Wrote chunk {} to {}", chunk.chunk_num, out_path.display());
    }
    Ok(())
}

fn main() -> Result<()> {
    let start_time = Instant::now();
    let args = Args::parse();

    if args.threads > 1024 {
        eprintln!("Warning: High thread count ({}). May impact performance.", args.threads);
    }
    if args.buffer_size > 10240 {
        eprintln!("Warning: Large buffer size ({}KB). May increase memory usage.", args.buffer_size);
    }

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
        process_streaming(file, file_size, &output_dir, &args, &split_dfa)?;
    } else {
        if args.verbose {
            println!("Using memory-mapped mode");
        }
        process_mapped(file, file_size, &output_dir, &args, &split_dfa)?;
    }

    let duration = start_time.elapsed();
    println!("Total processing time: {:.2?} seconds", duration);
    Ok(())
}

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
    let total_chunks = positions.len() - 1;
    let pb = ProgressBar::new(total_chunks as u64);
    let style = ProgressStyle::default_bar()
        .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({percent}%)")?
        .progress_chars("#>-");
    pb.set_style(style);

    process_chunks(&mmap, positions, output_dir, &counter, args, &pb)?;
    pb.finish_with_message("Processing complete");
    Ok(())
}

fn find_all_split_positions<A: Automaton>(
    file: &File,
    dfa: &A,
    segment_size: usize,
) -> Result<Vec<usize>> {
    let mut positions = Vec::new();
    let mut reader = BufReader::with_capacity(segment_size, file);
    let mut offset = 0;
    let mut at_line_start = true;

    loop {
        let mut buffer = vec![0u8; segment_size];
        let bytes_read = reader.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        buffer.truncate(bytes_read);
        let segment_positions = find_split_positions_in_segment(&buffer, dfa, at_line_start, offset);
        positions.extend(segment_positions);
        at_line_start = buffer.last().map_or(false, |&b| b == b'\n');
        offset += bytes_read;
    }
    positions.push(offset); // End of file as final position
    positions.sort_unstable();
    positions.dedup();
    Ok(positions)
}

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

fn process_chunks(
    data: &[u8],
    positions: Vec<usize>,
    output_dir: &PathBuf,
    counter: &AtomicUsize,
    args: &Args,
    pb: &ProgressBar,
) -> Result<()> {
    let chunks: Vec<_> = positions
        .windows(2)
        .map(|window| Chunk {
            start: window[0],
            end: window[1],
            chunk_num: counter.fetch_add(1, Ordering::Relaxed),
        })
        .collect();

    chunks.into_par_iter().try_for_each(|chunk| -> Result<()> {
        process_chunk(
            data,
            chunk,
            output_dir,
            args.trim,
            args.buffer_size,
            args.verbose,
        )?;
        pb.inc(1);
        Ok(())
    })?;

    if args.verbose {
        println!("Processed {} chunks", counter.load(Ordering::Relaxed));
    }
    Ok(())
}

fn process_streaming<A: Automaton + Send + Sync + 'static>(
    file: File,
    file_size: usize,
    output_dir: &PathBuf,
    args: &Args,
    split_dfa: &Arc<A>,
) -> Result<()> {
    let segment_size = args.segment_size * 1024 * 1024;
    let positions = find_all_split_positions(&file, &**split_dfa, segment_size)?;
    let total_chunks = positions.len() - 1;

    let pb = ProgressBar::new(total_chunks as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({percent}%)")?
            .progress_chars("#>-"),
    );

    let chunks: Vec<_> = positions
        .windows(2)
        .enumerate()
        .map(|(i, window)| Chunk {
            start: window[0],
            end: window[1],
            chunk_num: i,
        })
        .collect();

    chunks.into_par_iter().try_for_each(|chunk| -> Result<()> {
        let mut file = file.try_clone()?;
        file.seek(std::io::SeekFrom::Start(chunk.start as u64))?;
        let mut reader = BufReader::with_capacity(args.buffer_size * 1024, file);
        let mut data = vec![0u8; chunk.end - chunk.start];
        reader.read_exact(&mut data)?;
        process_chunk(
            &data,
            Chunk {
                start: 0,
                end: data.len(),
                chunk_num: chunk.chunk_num,
            },
            output_dir,
            args.trim,
            args.buffer_size,
            args.verbose,
        )?;
        pb.inc(1);
        Ok(())
    })?;

    pb.finish_with_message("Processing complete");
    Ok(())
}

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
