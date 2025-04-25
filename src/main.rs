use std::{
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Write, Seek},
    path::PathBuf,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Instant,
};

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

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "A CLI tool for splitting files based on a regex pattern.",
    long_about = "Splits large files into chunks based on a regex pattern, mimicking GNU csplit. Supports memory-mapped and streaming modes.\n\nUse --streaming for large files or to force streaming mode.\n\nTune performance with buffer_size, max_memory, and segment_size."
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

struct SharedBuffer {
    data: Arc<Vec<u8>>,
    positions: Vec<usize>,
    segment_offset: usize,
}

fn generate_filename(data: &[u8], chunk_num: usize, numbered: bool) -> String {
    if numbered {
        return format!("xx{:02}", chunk_num);
    }

    // Scan up to 10 lines or 1KB to find a line starting with '*'
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
    }.trim();

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

    // Ensure unique filenames by including chunk_num
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

    // Handle zero-byte chunks
    let chunk_data = &data[chunk.start..chunk.end];
    let filename = generate_filename(chunk_data, chunk.chunk_num, numbered);
    let out_path = output_dir.join(&filename);
    if chunk.start == chunk.end {
        std::fs::write(&out_path, "")?;
        if verbose {
            println!("Wrote empty chunk {} to {}", chunk.chunk_num, out_path.display());
        }
        return Ok(());
    }

    let file = File::create(&out_path)
        .with_context(|| format!("Failed to create file: {}", out_path.display()))?;

    // Use large buffer for large chunks
    let chunk_size = chunk.end - chunk.start;
    let buffer_capacity = if chunk_size > 1024 * 1024 * 1024 {
        buffer_size * 1024 * 16 // 16x for >1GB
    } else {
        buffer_size * 1024 * 4 // 4x default
    }.min(128 * 1024 * 1024); // Cap at 128MB

    let mut writer = BufWriter::with_capacity(buffer_capacity, file);
    let mut line_count = 0;
    let mut start = 0;
    const MAX_PROCESS_SIZE: usize = 64 * 1024 * 1024; // 64MB sub-chunk

    // Process large chunks in sub-chunks to manage memory
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
        // Flush less frequently for large chunks
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
    
    // Add start position only if this is the beginning of the file
    if segment_offset == 0 {
        positions.push(0); // Start of file
    }
    
    // Scan for split positions with a more efficient algorithm
    let mut i = 0;
    let mut line_start = 0;
    
    while i < data.len() {
        let byte = data[i];
        
        // Check for newline
        if byte == b'\n' {
            // Next position is start of a new line
            line_start = i + 1;
        } 
        // Only check for pattern at start of lines to improve performance
        else if i == line_start && i < data.len() {
            let input = Input::new(&data[i..]).anchored(Anchored::Yes);
            if matches!(dfa.try_search_fwd(&input), Ok(Some(_))) {
                positions.push(segment_offset + i);
            }
        }
        
        i += 1;
    }
    
    // Include the end point if we're at the end of the data
    if !positions.is_empty() && positions[positions.len() - 1] != segment_offset + data.len() {
        positions.push(segment_offset + data.len());
    }
    
    if verbose && !positions.is_empty() {
        println!(
            "Found {} split points in segment of {} bytes starting at {}",
            positions.len(),
            data.len(),
            segment_offset
        );
    }
    
    positions
}

fn process_mapped(
    file: File,
    _file_size: usize,
    output_dir: &PathBuf,
    args: &Args,
    split_dfa: &Arc<dense::DFA<Vec<u32>>>,
) -> Result<()> {
    let mmap = unsafe { Mmap::map(&file)? };
    let counter = AtomicUsize::new(0);

    let positions = find_split_positions(&mmap, &*split_dfa, 0, args.verbose);
    let total_chunks = positions.len().saturating_sub(1);
    let pb = ProgressBar::new(total_chunks as u64);
    let style = ProgressStyle::default_bar()
        .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} chunks ({percent}%)")?
        .progress_chars("#>-");
    pb.set_style(style);

    let chunks: Vec<_> = positions
        .windows(2)
        .map(|window| Chunk {
            start: window[0],
            end: window[1],
            chunk_num: counter.fetch_add(1, Ordering::Relaxed),
        })
        .collect();

    // Apply sequential advice if any chunk is large
    let has_large_chunk = chunks.iter().any(|chunk| chunk.end - chunk.start > 1024 * 1024 * 1024);
    if has_large_chunk {
        mmap.advise(memmap2::Advice::Sequential)?;
    }

    // Process chunks in parallel
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
    _file: File,
    file_size: usize,
    output_dir: &PathBuf,
    args: &Args,
    split_dfa: &Arc<dense::DFA<Vec<u32>>>,
) -> Result<()> {
    // Calculate segment size based on file size for better handling of very large files
    let segment_size = if file_size > 10 * 1024 * 1024 * 1024 { // > 10GB
        1024 * 1024 * 1024 // Use 1GB segments for huge files
    } else {
        (args.segment_size * 1024 * 1024).min(512 * 1024 * 1024) // Cap at 512MB for regular files
    };
    
    // Increase overlap size for large files to ensure pattern matches aren't missed
    let overlap_size = 256 * 1024; // 256KB overlap (increased from 64KB)
    let verbose = args.verbose;
    let trim = args.trim;
    let buffer_size = args.buffer_size;
    let numbered = args.numbered;
    let output_dir = output_dir.clone();
    let input_path = PathBuf::from(&args.input);

    if verbose {
        println!("Phase 1: Identifying all split positions with segment size: {} bytes, overlap: {} bytes", 
                 segment_size, overlap_size);
    }
    
    let pb = ProgressBar::new(file_size as u64);
    let style = ProgressStyle::default_bar()
        .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} bytes ({percent}%)")?
        .progress_chars("#>-");
    pb.set_style(style);
    
    // Use a reader to scan the entire file
    let file_reader = File::open(&input_path)?;
    let mut reader = BufReader::with_capacity(segment_size, file_reader);
    let mut positions = Vec::with_capacity(1024);
    positions.push(0); // Start position
    
    let mut absolute_pos = 0;
    let mut overlap_buffer = Vec::with_capacity(overlap_size * 2);
    
    // Track the last valid split position to prevent splitting very large chunks
    let mut last_valid_pos = 0;
    let max_chunk_size = if file_size > 20 * 1024 * 1024 * 1024 { // > 20GB
        file_size // Don't force splits on very large files
    } else {
        4 * 1024 * 1024 * 1024 // 4GB max chunk size for regular files
    };
    
    loop {
        let mut buffer = vec![0u8; segment_size];
        let bytes_read = reader.read(&mut buffer)?;
        if bytes_read == 0 {
            // End of file
            if !overlap_buffer.is_empty() {
                // Process final overlap buffer
                let segment_start = absolute_pos - overlap_buffer.len();
                let segment_positions = find_split_positions(
                    &overlap_buffer,
                    &*split_dfa,
                    segment_start,
                    verbose,
                );
                
                // Add positions from the last segment
                for pos in segment_positions {
                    if !positions.contains(&pos) && (pos - last_valid_pos <= max_chunk_size || pos == absolute_pos) {
                        positions.push(pos);
                        last_valid_pos = pos;
                    }
                }
            }
            
            // Always add end position if not already present
            if !positions.contains(&absolute_pos) {
                positions.push(absolute_pos);
            }
            break;
        }
        buffer.truncate(bytes_read);
        
        // Calculate segment offset for full buffer (including overlap)
        let segment_start = absolute_pos - overlap_buffer.len();
        
        // Combine overlap with new buffer with pre-allocated capacity
        let mut full_buffer = Vec::with_capacity(bytes_read + overlap_buffer.len());
        full_buffer.extend_from_slice(&overlap_buffer);
        full_buffer.extend_from_slice(&buffer);
        
        // Find split positions with absolute offsets
        let segment_positions = find_split_positions(
            &full_buffer,
            &*split_dfa,
            segment_start,
            verbose,
        );
        
        // Add positions from this segment with max chunk size constraint
        for pos in segment_positions {
            // Only add position if it's not already included and doesn't create chunks that are too large
            // This helps preserve large files that should be kept intact
            if !positions.contains(&pos) && (pos - last_valid_pos <= max_chunk_size || pos == absolute_pos) {
                positions.push(pos);
                last_valid_pos = pos;
            }
        }
        
        // Update overlap buffer
        overlap_buffer.clear();
        if bytes_read > overlap_size {
            overlap_buffer.extend_from_slice(&buffer[bytes_read - overlap_size..]);
        } else {
            overlap_buffer.extend_from_slice(&buffer);
        }
        
        absolute_pos += bytes_read;
        pb.set_position(absolute_pos as u64);
    }
    
    pb.finish_with_message("Split positions identified");
    
    // Sort positions to ensure they're in ascending order
    positions.sort_unstable();
    
    // Remove duplicate positions
    positions.dedup();
    
    // Analyze positions to ensure reasonable chunk sizes
    // This helps prevent excessive splitting of large files that match patterns
    let mut filtered_positions = Vec::with_capacity(positions.len());
    filtered_positions.push(0); // Always include start
    
    // Handle very large files (>1GB) specially - preserve them if they seem to be single logical units
    let contains_large_files = positions.windows(2).any(|window| window[1] - window[0] > 1024 * 1024 * 1024);
    
    if contains_large_files && verbose {
        println!("Large files detected - optimizing split points for large file handling");
    }
    
    // Add filtered positions based on file size
    for i in 1..positions.len() {
        let chunk_size = positions[i] - positions[i-1];
        let prev_pos = if filtered_positions.is_empty() { 0 } else { *filtered_positions.last().unwrap() };
        
        // Always include file end
        if i == positions.len() - 1 {
            filtered_positions.push(positions[i]);
            continue;
        }
        
        // Special handling for large chunks
        if chunk_size > 1024 * 1024 * 1024 { // > 1GB
            // Keep very large files intact if they seem to be a logical unit
            if chunk_size > 10 * 1024 * 1024 * 1024 { // > 10GB
                filtered_positions.push(positions[i]);
            } else if positions[i] - prev_pos > max_chunk_size {
                // Add this position if it would create a chunk larger than max_chunk_size
                filtered_positions.push(positions[i]);
            }
        } else {
            // For smaller chunks, use normal position
            filtered_positions.push(positions[i]);
        }
    }
    
    // Use filtered positions instead of raw positions
    positions = filtered_positions;
    
    if verbose {
        println!("Found {} split points in file after optimization", positions.len());
        
        // Log the size of each chunk for debugging
        for i in 1..positions.len() {
            let chunk_size = positions[i] - positions[i-1];
            if chunk_size > 1024 * 1024 * 1024 {
                println!("Chunk {}: {:.2} GB", i-1, chunk_size as f64 / (1024.0 * 1024.0 * 1024.0));
            } else if chunk_size > 1024 * 1024 {
                println!("Chunk {}: {:.2} MB", i-1, chunk_size as f64 / (1024.0 * 1024.0));
            }
        }
    }
    
    // Phase 2: Process each chunk based on the filtered split positions
    if verbose {
        println!("Phase 2: Processing chunks...");
    }
    
    let pb = ProgressBar::new((positions.len() - 1) as u64);
    let style = ProgressStyle::default_bar()
        .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} chunks ({percent}%)")?
        .progress_chars("#>-");
    pb.set_style(style);
    
    // Create a thread pool with limited threads for large files to avoid memory issues
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(if file_size > 10 * 1024 * 1024 * 1024 { 
            args.threads.min(4).max(1) // Limit threads for very large files
        } else {
            args.threads.min(num_cpus::get() * 2)
        })
        .build()?;
    
    // Process chunks with correct filenames based on positions
    let chunk_count = positions.len() - 1;
    let counter = AtomicUsize::new(0);
    
    let chunk_results: Vec<_> = pool.install(|| {
        (0..chunk_count).into_par_iter().map(|i| -> Result<()> {
            let start_pos = positions[i];
            let end_pos = positions[i + 1];
            let chunk_size = end_pos - start_pos;
            
            // Skip empty chunks
            if chunk_size == 0 {
                pb.inc(1);
                return Ok(());
            }
            
            // Get a unique chunk number using the atomic counter
            let chunk_num = counter.fetch_add(1, Ordering::Relaxed);
            
            // Open input file for each chunk to avoid file handle contention
            let file = File::open(&input_path)?;
            let mut reader = BufReader::with_capacity(buffer_size * 1024, file);
            
            // Seek to the start position
            reader.seek(std::io::SeekFrom::Start(start_pos as u64))?;
            
            // For very large chunks, use a different approach to avoid OOM
            if chunk_size > 4 * 1024 * 1024 * 1024 { // > 4GB
                if verbose {
                    println!("Processing large chunk {} ({:.2} GB) with streaming approach", 
                             chunk_num, chunk_size as f64 / (1024.0 * 1024.0 * 1024.0));
                }
                
                // Create output file directly
                let chunk_data_sample = {
                    // Read just a sample to determine the filename
                    let mut sample = vec![0u8; 4096];
                    let bytes_read = reader.read(&mut sample)?;
                    sample.truncate(bytes_read);
                    sample
                };
                
                // Generate filename from sample
                let filename = generate_filename(&chunk_data_sample, chunk_num, numbered);
                let out_path = output_dir.join(&filename);
                
                // Create output file
                let out_file = File::create(&out_path)?;
                let mut writer = BufWriter::with_capacity(buffer_size * 1024 * 4, out_file);
                
                // Re-seek to start to account for the sample we read
                reader.seek(std::io::SeekFrom::Start(start_pos as u64))?;
                
                // Process large file by streaming it in segments
                let mut line_count = 0;
                let mut buf = vec![0u8; buffer_size * 1024 * 8]; // Use larger buffer
                let mut total_written = 0;
                let mut total_read = 0;
                
                while total_read < chunk_size {
                    let to_read = buf.len().min(chunk_size - total_read);
                    let bytes_read = reader.read(&mut buf[0..to_read])?;
                    if bytes_read == 0 {
                        break; // End of file
                    }
                    
                    // Process the buffer line by line
                    let mut pos = 0;
                    while pos < bytes_read {
                        let line_end = buf[pos..bytes_read]
                            .iter()
                            .position(|&b| b == b'\n')
                            .map(|p| p + pos)
                            .unwrap_or(bytes_read);
                        
                        let line = &buf[pos..line_end];
                        if line_count >= 2 {
                            if line.len() > trim {
                                writer.write_all(&line[trim..])?;
                            } else {
                                writer.write_all(line)?;
                            }
                            if line_end < bytes_read {
                                writer.write_all(b"\n")?;
                            }
                            total_written += line.len() + 1;
                        }
                        line_count += 1;
                        pos = line_end + 1;
                    }
                    
                    total_read += bytes_read;
                    
                    // Periodically flush for very large files
                    if total_read % (64 * 1024 * 1024) == 0 {
                        writer.flush()?;
                    }
                }
                
                writer.flush()?;
                
                if verbose {
                    println!(
                        "Wrote large chunk {} to {} ({:.2} GB written)",
                        chunk_num,
                        out_path.display(),
                        total_written as f64 / (1024.0 * 1024.0 * 1024.0)
                    );
                }
            } else {
                // For normal sized chunks, use the original approach
                // Read the entire chunk
                let mut chunk_data = Vec::with_capacity(chunk_size);
                let mut buffer = vec![0u8; (buffer_size * 1024).min(chunk_size)];
                let mut total_read = 0;
                
                while total_read < chunk_size {
                    let to_read = buffer.len().min(chunk_size - total_read);
                    let bytes_read = reader.read(&mut buffer[0..to_read])?;
                    if bytes_read == 0 {
                        break; // End of file
                    }
                    
                    chunk_data.extend_from_slice(&buffer[0..bytes_read]);
                    total_read += bytes_read;
                }
                
                // Create a chunk with the correct chunk number for filename consistency
                let chunk = Chunk {
                    start: 0,
                    end: chunk_data.len(),
                    chunk_num,
                };
                
                // Use the standard process_chunk function to ensure consistent output
                process_chunk(
                    &chunk_data,
                    chunk,
                    &output_dir,
                    trim,
                    buffer_size,
                    verbose,
                    numbered,
                )?;
            }
            
            pb.inc(1);
            Ok(())
        }).collect()
    });
    
    // Check for any errors
    for result in chunk_results {
        result?;
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
        println!("Empty input file. Nothing to process.");
        return Ok(());
    }

    // Warn for large files
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
    let space_needed = (file_size as f64 * 1.1) as u64; // 10% extra
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
    
    // Use streaming mode for very large files (>4GB) regardless of max_memory setting
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