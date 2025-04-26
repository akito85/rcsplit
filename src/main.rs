use anyhow::{Context, Result};
use clap::Parser;
use fs4::available_space;
use indicatif::{ProgressBar, ProgressStyle};
use memmap2::Mmap;
use rayon::prelude::*;
use regex_automata::{
    dfa::{dense, Automaton},
    Anchored, Input,
};
use std::{
    fs::{self, File},
    io::{self, BufReader, BufWriter, BufRead, Read, Seek, SeekFrom, Write},
    path::PathBuf,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Instant,
};

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "A CLI tool for splitting files based on a regex pattern.",
    long_about = "Splits large files into chunks based on a regex pattern, mimicking GNU csplit. Ensures at least one chunk is created with filenames derived from lines starting with '*'. Supports memory-mapped and streaming modes.\n\nUse --streaming for large files or to force streaming mode.\n\nTune performance with buffer_size and max_memory."
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
    #[arg(long, default_value_t = false)]
    verbose: bool,
    #[arg(long, default_value_t = 0)]
    threads: usize,
    #[arg(long, default_value_t = 300)]
    timeout_secs: u64,
    #[arg(long, default_value_t = false)]
    numbered: bool,
}

#[derive(Debug, Clone)]
struct Chunk {
    start: usize,
    end: usize,
    chunk_num: usize,
}

fn generate_filename(data: &[u8], chunk_num: usize, numbered: bool) -> String {
    if numbered {
        return format!("xx{:02}", chunk_num);
    }

    let max_scan = data.len().min(1024);
    let mut lines = Vec::with_capacity(10);
    let mut start = 0;
    let mut line_count = 0;

    while start < max_scan && line_count < 10 {
        let line_end = data[start..]
            .iter()
            .position(|&b| b == b'\n')
            .map(|p| p + start)
            .unwrap_or(max_scan);
        lines.push(&data[start..line_end]);
        line_count += 1;
        start = line_end + 1;
    }

    let name_line = lines
        .iter()
        .find(|line| line.starts_with(b"*"))
        .copied()
        .unwrap_or(b"empty");

    let name_str = std::str::from_utf8(name_line).unwrap_or("invalid_utf8");
    let base_name = if name_str.starts_with('*') {
        &name_str[1..]
    } else {
        name_str
    }
    .trim();

    let sanitized = base_name
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
        .collect::<String>();
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
    numbered: bool,
) -> Result<()> {
    if chunk.start >= chunk.end || chunk.end > data.len() {
        if verbose {
            eprintln!("Skipping invalid chunk: {:?}", chunk);
        }
        return Ok(());
    }

    let chunk_data = &data[chunk.start..chunk.end];
    let filename = generate_filename(chunk_data, chunk.chunk_num, numbered);
    let out_path = output_dir.join(&filename);
    if chunk.start == chunk.end {
        fs::write(&out_path, "")?;
        if verbose {
            println!("Wrote empty chunk {} to {}", chunk.chunk_num, out_path.display());
        }
        return Ok(());
    }

    let file = File::create(&out_path)
        .with_context(|| format!("Failed to create file: {}", out_path.display()))?;
    let chunk_size = chunk.end - chunk.start;
    let buffer_capacity = if chunk_size > 1024 * 1024 * 1024 {
        buffer_size * 1024 * 16
    } else {
        buffer_size * 1024 * 4
    }
    .min(128 * 1024 * 1024);

    let mut writer = BufWriter::with_capacity(buffer_capacity, file);
    let mut line_count = 0;
    let mut start = 0;
    const MAX_PROCESS_SIZE: usize = 64 * 1024 * 1024;

    while start < chunk_data.len() {
        let process_end = (start + MAX_PROCESS_SIZE).min(chunk_data.len());
        let sub_chunk = &chunk_data[start..process_end];
        let mut pos = 0;

        while pos < sub_chunk.len() {
            let line_end = sub_chunk[pos..]
                .iter()
                .position(|&b| b == b'\n')
                .map(|p| p + pos)
                .unwrap_or(sub_chunk.len());
            let line = &sub_chunk[pos..line_end];
            if line_count >= 2 {
                if line.len() > trim {
                    writer.write_all(&line[trim..])?;
                } else {
                    writer.write_all(line)?;
                }
                if line_end < sub_chunk.len() {
                    writer.write_all(b"\n")?;
                }
            }
            line_count += 1;
            pos = line_end + 1;
        }
        start = process_end;
        if chunk_size > 1024 * 1024 * 1024 {
            writer.flush()?;
        }
    }

    writer.flush()?;
    if verbose {
        println!(
            "Wrote chunk {} to {} ({} bytes)",
            chunk.chunk_num,
            out_path.display(),
            chunk_data.len()
        );
    }
    Ok(())
}

fn find_split_positions(
    data: &[u8],
    dfa: &dense::DFA<Vec<u32>>,
    segment_offset: usize,
    verbose: bool,
) -> Vec<usize> {
    let mut positions = Vec::with_capacity(1024);
    positions.push(segment_offset);

    let mut i = 0;
    let mut line_start = 0;

    while i < data.len() {
        let byte = data[i];
        if byte == b'\n' {
            line_start = i + 1;
        } else if i == line_start && i < data.len() {
            let input = Input::new(&data[i..]).anchored(Anchored::Yes);
            if matches!(dfa.try_search_fwd(&input), Ok(Some(_))) {
                positions.push(segment_offset + i);
            }
        }
        i += 1;
    }

    positions.push(segment_offset + data.len());
    positions.sort_unstable();
    positions.dedup();

    if verbose && positions.len() > 2 {
        println!(
            "Found {} split points in segment of {} bytes starting at {}",
            positions.len() - 2,
            data.len(),
            segment_offset
        );
    }
    positions
}

fn process_mapped(
    file: File,
    file_size: usize,
    output_dir: &PathBuf,
    args: &Args,
    split_dfa: &Arc<dense::DFA<Vec<u32>>>,
) -> Result<()> {
    let mmap = unsafe { Mmap::map(&file)? };
    let counter = AtomicUsize::new(0);

    let positions = find_split_positions(&mmap, &split_dfa, 0, args.verbose);
    let total_chunks = positions.len().saturating_sub(1).max(1);
    let pb = ProgressBar::new(total_chunks as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} chunks ({percent}%)")?
            .progress_chars("#>-"),
    );

    let chunks: Vec<_> = if positions.len() <= 2 && positions[0] == 0 && positions[1] == file_size {
        vec![Chunk {
            start: 0,
            end: file_size,
            chunk_num: counter.fetch_add(1, Ordering::Relaxed),
        }]
    } else {
        positions
            .windows(2)
            .map(|window| Chunk {
                start: window[0],
                end: window[1],
                chunk_num: counter.fetch_add(1, Ordering::Relaxed),
            })
            .collect()
    };

    let has_large_chunk = chunks.iter().any(|chunk| chunk.end - chunk.start > 1024 * 1024 * 1024);
    if has_large_chunk {
        mmap.advise(memmap2::Advice::Sequential)?;
    }

    chunks.par_iter().try_for_each(|chunk| -> Result<()> {
        process_chunk(
            &mmap,
            chunk.clone(),
            output_dir,
            args.trim,
            args.buffer_size,
            args.verbose,
            args.numbered,
        )?;
        pb.inc(1);
        Ok(())
    })?;

    pb.finish_with_message("Processing complete");
    Ok(())
}

fn process_streaming(
    file: File,
    file_size: usize,
    output_dir: &PathBuf,
    args: &Args,
    split_dfa: &Arc<dense::DFA<Vec<u32>>>,
) -> Result<()> {
    let verbose = args.verbose;
    let input_path = PathBuf::from(&args.input);

    if verbose {
        println!("Phase 1: Identifying split positions in the file");
    }

    let pb = ProgressBar::new(file_size as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} bytes ({percent}%)")?
            .progress_chars("#>-"),
    );

    let mut reader = BufReader::with_capacity(8 * 1024 * 1024, File::open(&input_path)?);
    let mut positions = Vec::with_capacity(1024);
    positions.push(0);
    let mut buffer = Vec::with_capacity(16 * 1024);
    let mut pos = 0;
    let mut line_start = 0;

    while pos < file_size {
        buffer.clear();
        let bytes_read = reader.read_until(b'\n', &mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        if pos == line_start {
            let input = Input::new(&buffer).anchored(Anchored::Yes);
            if matches!(split_dfa.try_search_fwd(&input), Ok(Some(_))) {
                positions.push(pos);
                if verbose && positions.len() % 100 == 0 {
                    println!("Found {} split points so far", positions.len());
                }
            }
        }
        pos += bytes_read;
        line_start = pos;
        pb.set_position(pos as u64);
    }
    positions.push(file_size);
    positions.sort_unstable();
    positions.dedup();

    pb.finish_with_message(format!("Found {} split points", positions.len() - 2));

    if verbose {
        println!("Split positions: {:?}", positions);
    }

    if verbose {
        println!("Phase 2: Processing {} chunks", positions.len() - 1);
    }

    let chunks: Vec<_> = if positions.len() <= 2 && positions[0] == 0 && positions[1] == file_size {
        vec![(0, file_size)]
    } else {
        positions.windows(2).map(|w| (w[0], w[1])).collect()
    };

    let total_chunks = chunks.len().max(1);
    let pb = ProgressBar::new(total_chunks as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} chunks ({percent}%)")?
            .progress_chars("#>-"),
    );

    let counter = AtomicUsize::new(0);
    for (start_pos, end_pos) in chunks {
        let chunk_size = end_pos - start_pos;
        if chunk_size == 0 {
            if verbose {
                println!("Writing empty chunk {} at {}", counter.fetch_add(1, Ordering::Relaxed), start_pos);
            }
            let chunk_num = counter.fetch_add(1, Ordering::Relaxed);
            let filename = generate_filename(&[], chunk_num, args.numbered);
            let out_path = output_dir.join(&filename);
            fs::write(&out_path, "")?;
            continue;
        }

        let chunk_num = counter.fetch_add(1, Ordering::Relaxed);
        let mut reader = BufReader::with_capacity(8 * 1024 * 1024, File::open(&input_path)?);
        reader.seek(SeekFrom::Start(start_pos as u64))?;

        // Read up to 10 lines or 1KB for naming
        let mut sample = Vec::with_capacity(1024);
        let mut lines_read = 0;
        let mut total_bytes = 0;
        buffer.clear();
        while lines_read < 10 && total_bytes < 1024 && total_bytes < chunk_size {
            let bytes_read = reader.read_until(b'\n', &mut buffer)?;
            if bytes_read == 0 {
                break;
            }
            sample.extend_from_slice(&buffer[..bytes_read]);
            total_bytes += bytes_read;
            lines_read += 1;
            buffer.clear();
        }

        let filename = generate_filename(&sample, chunk_num, args.numbered);
        let out_path = output_dir.join(&filename);

        if verbose {
            println!(
                "Writing chunk {} to {} ({} bytes)",
                chunk_num, out_path.display(), chunk_size
            );
        }

        reader.seek(SeekFrom::Start(start_pos as u64))?;
        let file = File::create(&out_path)?;
        let mut writer = BufWriter::with_capacity(8 * 1024 * 1024, file);

        let mut bytes_processed = 0;
        let buffer_size = 8 * 1024 * 1024;
        let mut buffer = vec![0u8; buffer_size];

        while bytes_processed < chunk_size {
            let bytes_to_read = buffer_size.min(chunk_size - bytes_processed);
            let bytes_read = reader.read(&mut buffer[0..bytes_to_read])?;
            if bytes_read == 0 {
                eprintln!(
                    "Warning: Early EOF at chunk {} position {}",
                    chunk_num,
                    start_pos + bytes_processed
                );
                break;
            }
            writer.write_all(&buffer[0..bytes_read])?;
            bytes_processed += bytes_read;
        }
        writer.flush()?;
        pb.inc(1);
    }

    pb.finish_with_message("All chunks processed");
    Ok(())
}

fn main() -> Result<()> {
    let start_time = Instant::now();
    let args = Args::parse();

    if args.threads > 1024 {
        eprintln!("Warning: High thread count ({}). May impact performance.", args.threads);
    }
    if args.buffer_size > 10240 {
        eprintln!(
            "Warning: Large buffer size ({}KB). May increase memory usage.",
            args.buffer_size
        );
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
        return Err(anyhow::anyhow!(
            "Input file does not exist: {}",
            input_path.display()
        ));
    }

    let output_dir = PathBuf::from(&args.output);
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("Failed to create directory: {}", output_dir.display()))?;

    let file = File::open(&input_path)
        .with_context(|| format!("Failed to open file: {}", input_path.display()))?;
    let file_size = file.metadata()?.len() as usize;

    if file_size == 0 {
        let out_path = output_dir.join("chunk_00000.000");
        fs::write(&out_path, "")?;
        println!("Empty input file. Created empty chunk at {}", out_path.display());
        return Ok(());
    }

    if file_size as u64 > 4 * 1024 * 1024 * 1024 {
        eprintln!("Warning: Large files (>4GB) may not be supported on FAT32/exFAT.");
    }
    if file_size as u64 > i64::MAX as u64 {
        eprintln!(
            "Warning: File size ({} bytes) exceeds 2^63-1. Some operations may fail.",
            file_size
        );
    }

    let available_space = available_space(&output_dir).unwrap_or(u64::MAX);
    let space_needed = (file_size as f64 * 1.1) as u64;
    if available_space < space_needed {
        return Err(anyhow::anyhow!(
            "Insufficient disk space: {} bytes available, ~{} bytes required",
            available_space,
            space_needed
        ));
    }

    let split_dfa = Arc::new(
        dense::Builder::new()
            .configure(
                dense::Config::new()
                    .byte_classes(true)
                    .start_kind(regex_automata::dfa::StartKind::Anchored),
            )
            .build(&args.pattern)
            .map_err(|e| anyhow::anyhow!("Invalid regex pattern: {}", e))?,
    );

    if args.verbose {
        println!("Processing file: {} ({} bytes)", args.input, file_size);
    }

    let max_memory_bytes = args.max_memory * 1024 * 1024;
    let force_streaming = file_size > 4 * 1024 * 1024 * 1024;

    if args.streaming || file_size > max_memory_bytes || force_streaming {
        if args.verbose {
            if force_streaming {
                println!("Using streaming mode (forced for file > 4GB)");
            } else {
                println!("Using streaming mode");
            }
        }
        process_streaming(file, file_size, &output_dir, &args, &split_dfa)?;
    } else {
        if args.verbose {
            println!("Using memory-mapped mode");
        }
        process_mapped(file, file_size, &output_dir, &args, &split_dfa)?;
    }

    let duration = start_time.elapsed();
    println!("Total processing time: {:.2?}", duration);
    Ok(())
}