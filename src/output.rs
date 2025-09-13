use core::fmt;
use std::{
    collections::VecDeque,
    fs::File,
    io,
    io::{Read, Seek, SeekFrom, Write},
};

/// Wrapper over a file that calls [`FileExt::unlock`] at [dropping][`Drop`].
pub struct FileLock<'a>(pub &'a File);

impl<'a> FileLock<'a> {
    pub fn wrap_exclusive(f: &'a File) -> io::Result<Self> {
        f.lock()?;
        Ok(Self(f))
    }
}

impl<'a> core::ops::Deref for FileLock<'a> {
    type Target = &'a File;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a> core::ops::DerefMut for FileLock<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<'a> Drop for FileLock<'a> {
    fn drop(&mut self) {
        let _ = self.0.unlock();
    }
}

pub struct NowTime;

impl fmt::Display for NowTime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use chrono::{DateTime, SecondsFormat, Utc};
        let now = std::time::SystemTime::now();
        let now: DateTime<Utc> = now.into();
        write!(f, "{}", now.to_rfc3339_opts(SecondsFormat::Micros, true))
    }
}

pub struct LogDupler<Writer: io::Write> {
    pub writer: Writer,
    pub log_file: Option<File>,
    pub log_file_name: Option<String>,
    pub max_size: u64,
    pub unlogged_buffer: VecDeque<u8>,
}

impl<Writer: io::Write> LogDupler<Writer> {
    pub fn new(
        writer: Writer,
        log_file_name: Option<String>,
        max_size: u64,
        app_context: &str,
    ) -> Self {
        let mut initial_buf: Vec<u8> = Default::default();
        let _ = writeln!(
            &mut initial_buf,
            "{} logging started at {}",
            app_context, NowTime
        );
        Self {
            writer,
            log_file: None,
            log_file_name,
            max_size,
            unlogged_buffer: initial_buf.into(),
        }
    }
    fn try_open(&mut self) -> bool {
        if self.log_file.is_none() {
            if let Some(name) = &self.log_file_name {
                self.log_file = std::fs::OpenOptions::new()
                    .read(true)
                    .append(true)
                    .create(true)
                    .open(name)
                    .ok();
            }
        }
        self.log_file.is_some()
    }

    fn write_deq_start(&mut self, len: usize) {
        self.try_open();
        let mut kept_buf = Vec::<_>::new();

        if let Some(file) = &self.log_file {
            if let Ok(mut locked) = FileLock::wrap_exclusive(file) {
                let _ = locked.write_all(&self.unlogged_buffer.make_contiguous()[..len]);
                if let Ok(metadata) = locked.metadata() {
                    let current_len = metadata.len();
                    if current_len > self.max_size {
                        let cut_len = current_len - self.max_size / 2u64;
                        if locked.seek(SeekFrom::Start(cut_len)).is_ok()
                            && locked.read_to_end(&mut kept_buf).is_err()
                        {
                            kept_buf.clear();
                            let _ = locked.seek(SeekFrom::End(0));
                        }
                    }
                }
                drop(locked);
            }
        }
        self.unlogged_buffer.drain(..len);
        if kept_buf.is_empty() {
            return;
        }
        self.log_file = None;
        //trunctating on windows requires opening in non-append mode
        if let Ok(file_to_truncate) = std::fs::OpenOptions::new()
            .write(true)
            .open(self.log_file_name.as_ref().unwrap())
        {
            if file_to_truncate.set_len(0).is_err() {
                return;
            }
            drop(file_to_truncate);
        }
        self.try_open();
        if let Some(file) = &self.log_file {
            if let Ok(mut locked) = FileLock::wrap_exclusive(file) {
                if locked.rewind().is_ok()
                    && write!(locked, "... earlier log truncated at {}", NowTime).is_ok()
                {
                    let _ = {
                        if let Some(line_end_pos) = kept_buf.iter().position(|c| c == &b'\n') {
                            locked.write_all(&kept_buf[line_end_pos..])
                        } else {
                            writeln!(locked)
                        }
                    };
                }
            }
        }
    }
}

impl<Writer: io::Write> Drop for LogDupler<Writer> {
    fn drop(&mut self) {
        let mut final_buf: Vec<u8> = Default::default();
        let _ = writeln!(&mut final_buf, "logging finished at {}", NowTime);
        self.unlogged_buffer.extend(final_buf);
        let _ = self.flush();
        if let Some(file) = &self.log_file {
            if let Ok(locked) = FileLock::wrap_exclusive(file) {
                let _ = locked.sync_all();
            }
        }
    }
}
impl<Writer: io::Write> Write for LogDupler<Writer> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let _ = self.writer.write(buf);
        self.unlogged_buffer.extend(buf);
        if let Some(rev_position) = self.unlogged_buffer.iter().rev().position(|c| c == &b'\n') {
            self.write_deq_start(self.unlogged_buffer.len() - rev_position);
        }
        Ok(buf.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        let _ = self.writer.flush();
        self.write_deq_start(self.unlogged_buffer.len());
        Ok(())
    }
}
