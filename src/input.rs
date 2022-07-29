use mortal::terminal::Key;
use mortal::Event;

use std::io::Write;

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
        if !self.try_prepare_terminal() {
            //terminal input not available, treate it as empty input
            return Ok(ReaderEvent::Completed);
        }
        let terminal = self.terminal.as_ref().unwrap();
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
            Some(Event::Key(Key::Ctrl('d'))) => Ok(ReaderEvent::Canceled),

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
    pub fn set_pass_fail_accent_color(&mut self, failed: bool) {
        if crate::close::close_requested() {
            //color methods not available while closing on windows
            return;
        }
        use mortal::terminal::Color;
        if !self.try_prepare_terminal() {
            //terminal input not available, don't perform wait.
            return;
        }
        let terminal = self.terminal.as_ref().unwrap();
        let _ = terminal.set_fg(Some(match failed {
            false => Color::Green,
            true => Color::Red,
        }));
    }

    pub fn wait_any_key(&mut self) {
        if crate::close::close_requested() {
            //interaction methods not available while closing on windows
            let seconds_wait = 3;
            let mut out = std::io::stdout();
            if let Err(_) = write!(out, "Closing in {seconds_wait} seconds: ") {
                return;
            }
            for i in (1..=seconds_wait).rev() {
                if let Err(_) = write!(out, "{i}... ") {
                    return;
                }
                if let Err(_) = out.flush() {
                    return;
                }
                std::thread::sleep(std::time::Duration::from_secs(1));
            }
            println!();
            return;
        }

        if !self.try_prepare_terminal() {
            //terminal input not available, don't perform wait.
            return;
        }
        let terminal = self.terminal.as_ref().unwrap();
        loop {
            match terminal.read_event(Some(std::time::Duration::ZERO)) {
                Ok(None) => break,
                Err(_) => return,
                _ => continue,
            }
        }
        if terminal
            .write_str("  press any key to continue...")
            .is_err()
        {
            return;
        }
        loop {
            match terminal.read_event(None) {
                Err(_) => return,
                Ok(Some(Event::NoEvent)) => continue,
                _user_event => break,
            };
        }
        let _ignored = terminal.set_fg(None);
        let _ignored = terminal.write_str("\n");
    }
    fn try_prepare_terminal(&mut self) -> bool {
        if self.prepare_state.is_some() {
            return true;
        }
        let tried_terminal = mortal::Terminal::new();
        self.terminal = match tried_terminal {
            Ok(terminal) => Some(terminal),
            _ => return false,
        };
        let terminal = self.terminal.as_ref().unwrap();
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
            Err(_) => return false,
            Ok(state) => self.prepare_state = Some(state),
        }
        true
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
        if crate::close::close_requested() {
            return; //don't touch console methods while closing
        }
        if let Some(terminal) = &mut self.terminal {
            if let Some(state) = self.prepare_state.take() {
                let _ = terminal.set_fg(None);
                let _ = terminal.restore(state);
            }
        }
    }
}
