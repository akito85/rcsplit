use anyhow::{Context, Result};
use clap::Parser;
use fs4::available_space;
use indicatif::{ProgressBar, ProgressStyle};
use memchr::memchr;
use memmap2::Mmap;
use rayon::prelude::*;
use regex_automata::{
    dfa::{dense, Automaton},
    Anchored, Input,
};
use std::{
    fs::{self, File},
    io::{BufRead, BufReader, BufWriter, Seek, SeekFrom, Write},
    path::PathBuf,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Instant,
};

// Command-line arguments for file splitting
#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "A CLI tool for splitting files based on a regex pattern.",
    long_about = "Splits large files into chunks based on a regex pattern, mimicking GNU csplit. Ensures at least one chunk is created with filenames derived from lines starting with '*'. Supports memory-mapped and streaming modes.\n\nUse --streaming for large files or to force streaming mode.\n\nTune performance with buffer_size and max_memory."
)]
struct Args {
    #[arg(short, long, help = "Input file path")]
    input: String,
    #[arg(short, long, help = "Regex pattern for splitting (e.g., '^\\*')")]
    pattern: String,
    #[arg(short, long, default_value = "out_chunks", help = "Output directory for chunks")]
    output: String,
    #[arg(short, long, default_value_t = 4, help = "Number of characters to trim from each line")]
    trim: usize,
    #[arg(
        short,
        long,
        default_value_t = 256,
        help = "Buffer size in KB for I/O operations"
    )]
    buffer_size: usize,
    #[arg(
        short,
        long,
        default_value_t = 512,
        help = "Maximum memory in MB for memory-mapped mode"
    )]
    max_memory: usize,
    #[arg(short, long, help = "Force streaming mode for large files")]
    streaming: bool,
    #[arg(long, default_value_t = false, help = "Enable verbose output (equivalent to --debug-level 1)")]
    verbose: bool,
    #[arg(
        long,
        default_value_t = 0,
        help = "Debug logging level (0=off, 1=basic, 2=actions, 3=names+lines, 4=trim+lines, 5=stats)"
    )]
    debug_level: u8,
    #[arg(long, default_value_t = 0, help = "Number of threads (0 = auto)")]
    threads: usize,
    #[arg(long, default_value_t = false, help = "Use numbered filenames (xxNN)")]
    numbered: bool,
}

// Struct to represent a chunk of the input file
#[derive(Debug, Clone)]
struct Chunk {
    start: usize,    // Start byte position
    end: usize,      // End byte position
    chunk_num: usize, // Chunk index
}

// Log debug messages based on the required debug level
// Levels: 1=basic, 2=actions, 3=names+lines, 4=trim+lines, 5=stats
fn debug_log(level: u8, required_level: u8, message: &str) {
    if level >= required_level && required_level > 0 {
        eprintln!("[DEBUG L{}] {}", required_level, message);
    }
}

// Generate filename from chunk data or chunk number
fn generate_filename(
    data: &[u8],
    chunk_num: usize,
    numbered: bool,
    debug_level: u8,
) -> String {
    if numbered {
        let filename = format!("xx{:02}", chunk_num);
        debug_log(debug_level, 1, &format!("Generated filename: {}", filename));
        return filename;
    }

    // For small chunks, use entire data; otherwise cap at 1KB or 10 lines
    let max_scan = if data.len() < 1024 { data.len() } else { 1024 };
    let mut lines = Vec::with_capacity(10);
    let mut start = 0;
    let mut line_count = 0;

    debug_log(
        debug_level,
        2,
        &format!("Sampling chunk {}: {} bytes", chunk_num, max_scan),
    );

    // Split into lines
    while start < max_scan && line_count < 10 {
        let line_end = memchr(b'\n', &data[start..])
            .map(|p| p + start)
            .unwrap_or(data.len().min(max_scan));
        let line = &data[start..line_end];
        if !line.is_empty() {
            lines.push(line);
        }
        line_count += 1;
        start = line_end + 1;
    }

    // Check if chunk starts with '*' for small chunks
    let name_line = if data.len() <= 20 && data.starts_with(b"*") {
        data
    } else {
        lines
            .iter()
            .find(|line| line.starts_with(b"*") && line.len() > 1)
            .copied()
            .unwrap_or(b"chunk")
    };

    // Convert to string, strip '*' prefix, trim whitespace
    let name_str = std::str::from_utf8(name_line).unwrap_or("chunk");
    let base_name = name_str.strip_prefix('*').unwrap_or(name_str).trim();

    if debug_level >= 3 {
        debug_log(
            debug_level,
            3,
            &format!("Chunk {}: Raw name {}, base {}", chunk_num, name_str, base_name),
        );
        for (i, line) in lines.iter().enumerate() {
            let s = String::from_utf8_lossy(line).into_owned();
            let s = if s.len() > 50 { format!("{}...", &s[..47]) } else { s };
            debug_log(debug_level, 3, &format!("Sample line {}: {}", i, s));
        }
    }

    // Sanitize: keep alphanumeric, '_', '-', replace others with '_'
    let sanitized: String = base_name
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
        .collect();
    let trimmed = sanitized.trim_matches('_');
    let final_base = if trimmed.is_empty() { "chunk" } else { trimmed };

    debug_log(
        debug_level,
        3,
        &format!(
            "Chunk {}: Sanitized name {}, final {}",
            chunk_num, sanitized, final_base
        ),
    );

    // Format filename
    let filename = format!("{}_{:05}.000", final_base, chunk_num);
    debug_log(
        debug_level,
        1,
        &format!("Generated filename: {} for chunk {}", filename, chunk_num),
    );

    filename
}

// Trim a chunk's lines (skip first two, trim 'trim' chars)
fn trim_chunk(
    reader: &mut BufReader<File>,
    chunk_size: usize,
    trim: usize,
    debug_level: u8,
) -> Result<(Vec<u8>, Option<String>, Option<String>)> {
    let mut output = Vec::with_capacity(chunk_size.min(1024 * 1024));
    let mut line_buffer = Vec::with_capacity(1024);
    let mut line_count = 0;
    let mut bytes_processed = 0;
    let mut sample_lines = Vec::new(); // Up to 3 trimmed lines
    let mut first_line = None;
    let mut second_line = None;

    debug_log(
        debug_level,
        1,
        &format!("Trimming chunk: {} bytes", chunk_size),
    );

    while bytes_processed < chunk_size {
        line_buffer.clear();
        let bytes_read = reader.read_until(b'\n', &mut line_buffer)?;
        if bytes_read == 0 {
            if bytes_processed < chunk_size {
                debug_log(
                    debug_level,
                    1,
                    &format!("Warning: Early EOF at {} bytes", bytes_processed),
                );
            }
            break;
        }

        bytes_processed += bytes_read;
        if bytes_processed > chunk_size {
            let excess = bytes_processed - chunk_size;
            line_buffer.truncate(bytes_read - excess);
        }

        let line = &line_buffer[..line_buffer.len().min(bytes_read)];
        if line_count == 0 && !line.is_empty() {
            let s = String::from_utf8_lossy(line).into_owned();
            first_line = Some(if s.len() > 50 { format!("{}...", &s[..47]) } else { s });
        } else if line_count == 1 && !line.is_empty() {
            let s = String::from_utf8_lossy(line).into_owned();
            second_line = Some(if s.len() > 50 { format!("{}...", &s[..47]) } else { s });
        }

        if line_count >= 2 {
            let trimmed = if line.len() > trim { &line[trim..] } else { b"" };
            output.extend_from_slice(trimmed);
            if bytes_processed < chunk_size || line.ends_with(b"\n") {
                output.push(b'\n');
            }
            if debug_level >= 4 && sample_lines.len() < 3 && !trimmed.is_empty() {
                let mut s = String::from_utf8_lossy(trimmed).into_owned();
                if s.len() > 50 {
                    s.truncate(47);
                    s.push_str("...");
                }
                sample_lines.push(s);
            }
        }
        line_count += 1;
    }

    debug_log(
        debug_level,
        1,
        &format!("Trimmed {} bytes from {} lines", output.len(), line_count),
    );

    if debug_level >= 3 && line_count > 0 {
        debug_log(
            debug_level,
            3,
            &format!(
                "Chunk: First line: {}, Second line: {}, Size before trim: {} bytes",
                first_line.as_ref().unwrap_or(&"None".to_string()),
                second_line.as_ref().unwrap_or(&"None".to_string()),
                chunk_size
            ),
        );
    }

    if debug_level >= 4 && line_count > 0 {
        debug_log(
            debug_level,
            4,
            &format!(
                "Chunk trimming: Processed {} lines, trimmed {} bytes, First line: {}, Second line: {}, Size after trim: {} bytes, sample: {:?}",
                line_count,
                output.len(),
                first_line.as_ref().unwrap_or(&"None".to_string()),
                second_line.as_ref().unwrap_or(&"None".to_string()),
                output.len(),
                sample_lines
            ),
        );
    }

    Ok((output, first_line, second_line))
}

// Process a single chunk, writing to a file
fn process_chunk(
    data: &[u8],
    chunk: Chunk,
    output_dir: &PathBuf,
    trim: usize,
    buffer_size: usize,
    debug_level: u8,
    numbered: bool,
) -> Result<usize> {
    if chunk.start >= chunk.end || chunk.end > data.len() {
        debug_log(
            debug_level,
            1,
            &format!("Skipped invalid chunk {}", chunk.chunk_num),
        );
        return Ok(0);
    }

    let chunk_data = &data[chunk.start..chunk.end];
    let filename = generate_filename(chunk_data, chunk.chunk_num, numbered, debug_level);
    let out_path = output_dir.join(&filename);

    if chunk.start == chunk.end {
        fs::write(&out_path, "")?;
        debug_log(
            debug_level,
            1,
            &format!("Wrote empty chunk {}: {}", chunk.chunk_num, filename),
        );
        return Ok(0);
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
    const WRITE_BATCH_SIZE: usize = 1024 * 1024;
    let mut write_buffer = Vec::with_capacity(WRITE_BATCH_SIZE);
    let mut sample_lines = Vec::new();
    let mut first_line = None;
    let mut second_line = None;

    debug_log(
        debug_level,
        1,
        &format!("Wrote chunk {}: {}, {} bytes", chunk.chunk_num, filename, chunk_size),
    );

    while start < chunk_data.len() {
        let process_end = (start + MAX_PROCESS_SIZE).min(chunk_data.len());
        let sub_chunk = &chunk_data[start..process_end];
        let mut pos = 0;

        while pos < sub_chunk.len() {
            let line_end = memchr(b'\n', &sub_chunk[pos..])
                .map(|p| p + pos)
                .unwrap_or(sub_chunk.len());
            let line = &sub_chunk[pos..line_end];
            if line_count == 0 && !line.is_empty() {
                let s = String::from_utf8_lossy(line).into_owned();
                first_line = Some(if s.len() > 50 { format!("{}...", &s[..47]) } else { s });
            } else if line_count == 1 && !line.is_empty() {
                let s = String::from_utf8_lossy(line).into_owned();
                second_line = Some(if s.len() > 50 { format!("{}...", &s[..47]) } else { s });
            }
            if line_count >= 2 {
                let trimmed = if line.len() > trim { &line[trim..] } else { b"" };
                write_buffer.extend_from_slice(trimmed);
                if line_end < sub_chunk.len() || start + line_end < chunk_data.len() {
                    write_buffer.push(b'\n');
                }
                if debug_level >= 4 && sample_lines.len() < 3 && !trimmed.is_empty() {
                    let mut s = String::from_utf8_lossy(trimmed).into_owned();
                    if s.len() > 50 {
                        s.truncate(47);
                        s.push_str("...");
                    }
                    sample_lines.push(s);
                }
            }
            line_count += 1;
            pos = line_end + 1;

            if write_buffer.len() >= WRITE_BATCH_SIZE {
                writer.write_all(&write_buffer)?;
                write_buffer.clear();
            }
        }
        start = process_end;

        if chunk_size > 1024 * 1024 * 1024 {
            writer.write_all(&write_buffer)?;
            write_buffer.clear();
            writer.flush()?;
        }
    }

    if !write_buffer.is_empty() {
        writer.write_all(&write_buffer)?;
    }
    writer.flush()?;

    if debug_level >= 3 && line_count > 0 {
        debug_log(
            debug_level,
            3,
            &format!(
                "Chunk {}: First line: {}, Second line: {}, Size before trim: {} bytes",
                chunk.chunk_num,
                first_line.as_ref().unwrap_or(&"None".to_string()),
                second_line.as_ref().unwrap_or(&"None".to_string()),
                chunk_size
            ),
        );
    }

    if debug_level >= 4 && line_count > 0 {
        debug_log(
            debug_level,
            4,
            &format!(
                "Chunk {} trimming: Processed {} lines, trimmed {} bytes, First line: {}, Second line: {}, Size after trim: {} bytes, sample: {:?}",
                chunk.chunk_num,
                line_count,
                chunk_data.len(),
                first_line.as_ref().unwrap_or(&"None".to_string()),
                second_line.as_ref().unwrap_or(&"None".to_string()),
                chunk_data.len(),
                sample_lines
            ),
        );
    }

    Ok(chunk_size)
}

// Find split positions in data using DFA regex
fn find_split_positions(
    data: &[u8],
    dfa: &dense::DFA<Vec<u32>>,
    segment_offset: usize,
    debug_level: u8,
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
                debug_log(
                    debug_level,
                    2,
                    &format!("Found split at position {}", segment_offset + i),
                );
            }
        }
        i += 1;
    }

    positions.push(segment_offset + data.len());
    positions.sort_unstable();
    positions.dedup();

    debug_log(
        debug_level,
        1,
        &format!("Found {} split points", positions.len() - 2),
    );

    positions
}

// Process file using memory-mapped I/O
fn process_mapped(
    file: File,
    file_size: usize,
    output_dir: &PathBuf,
    args: &Args,
    split_dfa: &Arc<dense::DFA<Vec<u32>>>,
) -> Result<usize> {
    let mmap = unsafe { Mmap::map(&file)? };
    let chunk_counter = AtomicUsize::new(0);
    let debug_level = if args.verbose { args.debug_level.max(1) } else { args.debug_level };

    debug_log(debug_level, 1, "Phase 1: Identifying split positions");
    let positions = find_split_positions(&mmap, split_dfa, 0, debug_level);
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
            chunk_num: chunk_counter.fetch_add(1, Ordering::Relaxed),
        }]
    } else {
        positions
            .windows(2)
            .map(|window| Chunk {
                start: window[0],
                end: window[1],
                chunk_num: chunk_counter.fetch_add(1, Ordering::Relaxed),
            })
            .collect()
    };

    debug_log(debug_level, 1, &format!("Split positions: {:?}", positions));

    let has_large_chunk = chunks.iter().any(|chunk| chunk.end - chunk.start > 1024 * 1024 * 1024);
    if has_large_chunk {
        mmap.advise(memmap2::Advice::Sequential)?;
    }

    debug_log(debug_level, 1, &format!("Phase 2: Processing {} chunks", total_chunks));

    let total_bytes: usize = chunks
        .par_iter()
        .map(|chunk| {
            process_chunk(
                &mmap,
                chunk.clone(),
                output_dir,
                args.trim,
                args.buffer_size,
                debug_level,
                args.numbered,
            )
            .unwrap_or(0)
        })
        .sum();

    pb.finish();
    Ok(total_bytes)
}

// Process file using streaming I/O
fn process_streaming(
    _file: File,
    file_size: usize,
    output_dir: &PathBuf,
    args: &Args,
    split_dfa: &Arc<dense::DFA<Vec<u32>>>,
) -> Result<usize> {
    let debug_level = if args.verbose { args.debug_level.max(1) } else { args.debug_level };
    let input_path = PathBuf::from(&args.input);

    debug_log(debug_level, 1, "Phase 1: Identifying split positions");

    let pb = ProgressBar::new(file_size as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} bytes ({percent}%)")?
            .progress_chars("#>-"),
    );

    let mut reader = BufReader::with_capacity(16 * 1024 * 1024, File::open(&input_path)?);
    let mut positions = Vec::with_capacity(1024);
    positions.push(0);
    let mut buffer = Vec::with_capacity(16 * 1024);
    let mut file_pos = 0;
    let mut line_start = 0;

    while file_pos < file_size {
        buffer.clear();
        let bytes_read = reader.read_until(b'\n', &mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        if file_pos == line_start {
            let input = Input::new(&buffer).anchored(Anchored::Yes);
            if matches!(split_dfa.try_search_fwd(&input), Ok(Some(_))) {
                positions.push(file_pos);
                debug_log(
                    debug_level,
                    2,
                    &format!("Found split at position {}", file_pos),
                );
            }
        }
        file_pos += bytes_read;
        line_start = file_pos;
        pb.set_position(file_pos as u64);
    }
    positions.push(file_size);
    positions.sort_unstable();
    positions.dedup();

    pb.finish();

    debug_log(debug_level, 1, &format!("Split positions: {:?}", positions));

    debug_log(debug_level, 1, &format!("Phase 2: Processing {} chunks", positions.len() - 1));

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

    let chunk_counter = AtomicUsize::new(0);
    let mut total_bytes = 0;

    for (start_pos, end_pos) in chunks {
        let chunk_size = end_pos - start_pos;
        let chunk_num = chunk_counter.fetch_add(1, Ordering::Relaxed);

        let mut reader = BufReader::with_capacity(16 * 1024 * 1024, File::open(&input_path)?);
        reader.seek(SeekFrom::Start(start_pos as u64))?;

        debug_log(
            debug_level,
            3,
            &format!("Seeking to position {} for chunk {}", start_pos, chunk_num),
        );

        // Sample chunk
        let mut sample = Vec::with_capacity(1024);
        let max_sample = if chunk_size < 1024 { chunk_size } else { 1024 };
        let mut total_bytes_sample = 0;
        let mut lines_read = 0;

        debug_log(
            debug_level,
            2,
            &format!(
                "Sampling chunk {}: {} bytes from position {}",
                chunk_num, max_sample, start_pos
            ),
        );

        while total_bytes_sample < max_sample && lines_read < 10 {
            buffer.clear();
            let bytes_read = reader.read_until(b'\n', &mut buffer)?;
            if bytes_read == 0 {
                break;
            }
            let to_copy = (max_sample - total_bytes_sample).min(bytes_read);
            sample.extend_from_slice(&buffer[..to_copy]);
            total_bytes_sample += to_copy;
            lines_read += 1;

            if debug_level >= 3 {
                let s = String::from_utf8_lossy(&buffer[..to_copy]).into_owned();
                let s = if s.len() > 50 { format!("{}...", &s[..47]) } else { s };
                debug_log(debug_level, 3, &format!("Sample: {}", s));
            }
        }

        let filename = generate_filename(&sample, chunk_num, args.numbered, debug_level);
        let out_path = output_dir.join(&filename);

        reader.seek(SeekFrom::Start(start_pos as u64))?;
        let file = File::create(&out_path)?;
        let mut writer = BufWriter::with_capacity(32 * 1024 * 1024, file);

        if chunk_size == 0 {
            fs::write(&out_path, "")?;
            debug_log(
                debug_level,
                1,
                &format!("Wrote empty chunk {}: {}", chunk_num, filename),
            );
            writer.flush()?;
            pb.inc(1);
            continue;
        }

        let (trimmed_data, _first_line, _second_line) = trim_chunk(&mut reader, chunk_size, args.trim, debug_level)?;
        writer.write_all(&trimmed_data)?;
        writer.flush()?;

        total_bytes += chunk_size;
        pb.inc(1);
    }

    pb.finish();
    Ok(total_bytes)
}

// Main function to orchestrate file splitting
fn main() -> Result<()> {
    let start_time = Instant::now();
    let args = Args::parse();
    let debug_level = if args.verbose { args.debug_level.max(1) } else { args.debug_level };

    let thread_count = if args.threads > 0 {
        args.threads.min(num_cpus::get().max(1))
    } else {
        num_cpus::get().max(1)
    };

    if thread_count > 1024 {
        debug_log(
            debug_level,
            1,
            &format!("Warning: High thread count ({}) may impact performance", thread_count),
        );
    }
    if args.buffer_size > 10240 {
        debug_log(
            debug_level,
            1,
            &format!(
                "Warning: Large buffer size ({}KB) may increase memory usage",
                args.buffer_size
            ),
        );
    }

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
        debug_log(
            debug_level,
            1,
            "Warning: Large files (>4GB) may not be supported on FAT32/exFAT",
        );
    }
    if file_size as u64 > i64::MAX as u64 {
        debug_log(
            debug_level,
            1,
            &format!(
                "Warning: File size ({} bytes) exceeds 2^63-1. Some operations may fail",
                file_size
            ),
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

    debug_log(
        debug_level,
        1,
        &format!("Processing file: {} ({} bytes)", args.input, file_size),
    );

    let max_memory_bytes = args.max_memory * 1024 * 1024;
    let force_streaming = file_size > 4 * 1024 * 1024 * 1024;
    let total_bytes;

    if args.streaming || file_size > max_memory_bytes || force_streaming {
        debug_log(
            debug_level,
            1,
            if force_streaming {
                "Using streaming mode (forced for file > 4GB)"
            } else {
                "Using streaming mode"
            },
        );
        total_bytes = process_streaming(file, file_size, &output_dir, &args, &split_dfa)?;
    } else {
        debug_log(debug_level, 1, "Using memory-mapped mode");
        total_bytes = process_mapped(file, file_size, &output_dir, &args, &split_dfa)?;
    }

    let duration = start_time.elapsed();
    let total_chunks = (total_bytes as f64 / file_size as f64).ceil() as usize;
    let peak_memory_mb = if args.streaming || file_size > max_memory_bytes || force_streaming {
        49.0 // BufReader (16MB) + BufWriter (32MB) + sample (1MB)
    } else {
        (file_size as f64 / 1024.0 / 1024.0).min(max_memory_bytes as f64 / 1024.0 / 1024.0) + 64.0
    };

    if debug_level >= 5 {
        debug_log(
            debug_level,
            5,
            &format!(
                "Summary: Processed {} ({} bytes), {} chunks, {} bytes written, time: {:.2?}, peak memory: ~{}MB",
                args.input, file_size, total_chunks, total_bytes, duration, peak_memory_mb as usize
            ),
        );
    }

    println!("Total processing time: {:.2?}", duration);
    Ok(())
}