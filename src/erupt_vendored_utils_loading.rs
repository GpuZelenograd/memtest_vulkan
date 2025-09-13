//! Load functions using [`libloading`](https://crates.io/crates/libloading).
//!
//! Enabled using the `loading` cargo feature.
use crate::{CustomEntryLoader, LoaderError};
use std::{error::Error, ffi::CStr, fmt};

#[cfg(all(
    unix,
    not(any(target_os = "macos", target_os = "ios", target_os = "android"))
))]
const LIB_PATH: &str = "libvulkan.so.1";

#[cfg(target_os = "android")]
const LIB_PATH: &str = "libvulkan.so";

#[cfg(any(target_os = "macos", target_os = "ios"))]
const LIB_PATH: &str = "libvulkan.dylib";

#[cfg(windows)]
const LIB_PATH: &str = "vulkan-1.dll";

pub type EntryLoader = CustomEntryLoader<libloading::Library>;

/// An error that can occur while initializing a [`EntryLoader`].
#[derive(Debug)]
pub enum EntryLoaderError {
    /// The library failed to load with a message about library purpose.
    Library(libloading::Error, String),
    /// The entry loader failed to initialize.
    EntryLoad(LoaderError),
}

impl fmt::Display for EntryLoaderError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            EntryLoaderError::Library(_, msg) => write!(f, "{msg}"),
            EntryLoaderError::EntryLoad(_) => write!(f, "The entry loader failed to initialize"),
        }
    }
}

impl Error for EntryLoaderError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            EntryLoaderError::Library(err, _) => Some(err),
            EntryLoaderError::EntryLoad(err) => Some(err),
        }
    }
}

pub fn new_loader() -> Result<EntryLoader, EntryLoaderError> {
    new_loader_with_lib_path(LIB_PATH)
}

pub fn new_loader_with_lib_path(lib_path: &str) -> Result<EntryLoader, EntryLoaderError> {
    let library = unsafe {
        libloading::Library::new(lib_path).map_err(|err| EntryLoaderError::Library(err, format!("Khronos 'vulkan loader' vendor-agnostic dynamic library `{lib_path}` failed to load")))?
    };

    let symbol = |library: &mut libloading::Library, name| unsafe {
        let cstr = CStr::from_ptr(name);
        let bytes = cstr.to_bytes_with_nul();
        library.get(bytes).ok().map(|symbol| *symbol)
    };

    unsafe { CustomEntryLoader::with_library(library, symbol).map_err(EntryLoaderError::EntryLoad) }
}
