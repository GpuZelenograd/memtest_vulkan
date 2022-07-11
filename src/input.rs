use mortal::terminal::Key;
use mortal::Event;

#[derive(Default)]
pub struct Reader {
    pub current_input: String,
    terminal: Option<mortal::Terminal>,
    prepare_state: Option<mortal::terminal::PrepareState>,
}

pub enum ReaderEvent {
    Timeout,
    Edited,
    Completed,
    Canceled,
}

impl Reader {
    pub fn input_digit_step(
        &mut self,
        prompt: &str,
        timeout: &std::time::Duration,
    ) -> Result<ReaderEvent, Box<dyn std::error::Error>> {
        let terminal;
        if self.prepare_state.is_none() {
            self.terminal = Some(mortal::Terminal::new()?);
            terminal = self.terminal.as_ref().unwrap();
            let mut signals = mortal::signal::SignalSet::new();
            signals.insert(mortal::signal::Signal::Break);
            signals.insert(mortal::signal::Signal::Interrupt);
            signals.insert(mortal::signal::Signal::Quit);
            let tried_prepare = terminal.prepare(mortal::terminal::PrepareConfig {
                block_signals: false,
                enable_keypad: false,
                report_signals: signals,
                ..mortal::terminal::PrepareConfig::default()
            });
            match tried_prepare {
                //terminal input not available, treate it as empty input
                Err(_) => return Ok(ReaderEvent::Completed),
                Ok(state) => self.prepare_state = Some(state),
            }
        } else {
            terminal = self.terminal.as_ref().unwrap();
        }
        terminal.move_to_first_column()?;
        terminal.clear_to_line_end()?;
        terminal.write_str(prompt)?;
        terminal.write_str(&self.current_input)?;
        match terminal.read_event(Some(*timeout))? {
            Some(Event::Key(Key::Enter)) => Ok(ReaderEvent::Completed),
            Some(Event::Key(Key::Ctrl('m'))) => Ok(ReaderEvent::Completed),
            Some(Event::Key(Key::Ctrl('j'))) => Ok(ReaderEvent::Completed),

            Some(Event::Key(Key::Escape)) => Ok(ReaderEvent::Canceled),
            Some(Event::Key(Key::Ctrl('q'))) => Ok(ReaderEvent::Canceled),
            Some(Event::Key(Key::Ctrl('c'))) => Ok(ReaderEvent::Canceled),
            Some(Event::Key(Key::Ctrl('z'))) => Ok(ReaderEvent::Canceled),
            Some(Event::Key(Key::Ctrl('x'))) => Ok(ReaderEvent::Canceled),

            Some(Event::Key(Key::Char(c))) => {
                if c.is_ascii_control() {
                    //unexpected control keys
                    Ok(ReaderEvent::Canceled)
                } else {
                    if c.is_ascii_digit() {
                        terminal.write_char(c)?;
                        self.current_input.push(c);
                    }
                    Ok(ReaderEvent::Edited)
                }
            }
            Some(Event::Key(Key::Backspace)) | Some(Event::Key(Key::Delete)) => {
                self.handle_backspace()?;
                Ok(ReaderEvent::Edited)
            }

            Some(Event::Key(_)) => Ok(ReaderEvent::Edited),
            Some(Event::Signal(_)) => Ok(ReaderEvent::Canceled),
            _ => Ok(ReaderEvent::Timeout),
        }
    }
    fn handle_backspace(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if !self.current_input.is_empty() {
            if let Some(terminal) = &mut self.terminal {
                terminal.move_left(1)?;
                terminal.clear_to_line_end()?;
            }
            self.current_input.pop();
        }
        Ok(())
    }
}

impl Drop for Reader {
    fn drop(&mut self) {
        if let Some(terminal) = &mut self.terminal {
            if let Some(state) = self.prepare_state.take() {
                terminal.restore(state).unwrap();
            }
        }
    }
}
