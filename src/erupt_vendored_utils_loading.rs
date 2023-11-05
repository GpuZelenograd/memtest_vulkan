//! Load functions using [`libloading`](https://crates.io/crates/libloading).
//!
//! Enabled using the `loading` cargo feature.
use crate::{CustomEntryLoader, LoaderError};
use libloading::Library;
use std::{
    error::Error,
    ffi::{CStr, OsStr},
    fmt::{self, Display},
};

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

/// The default entry loader library, used in [`EntryLoader`].
pub type DefaultEntryLoaderLibrary = Library;

/// The default [`EntryLoader`], providing `EntryLoader::new`.
pub type EntryLoader = CustomEntryLoader<DefaultEntryLoaderLibrary>;
pub struct EntryLoaderImpl(EntryLoader);

/// An error that can occur while loading a library.
#[derive(Debug)]
pub struct LibraryError(libloading::Error);

impl Display for LibraryError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Display::fmt(&self.0, f)
    }
}

impl Error for LibraryError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&self.0)
    }
}

/// An error that can occur while initializing a [`EntryLoader`].
#[derive(Debug)]
pub enum EntryLoaderError {
    /// The library failed to load.
    Library(LibraryError),
    /// The entry loader failed to initialize.
    EntryLoad(LoaderError),
}

impl Display for EntryLoaderError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            EntryLoaderError::Library(_) => write!(f, "GPU-vendor-agnostic Khronos 'vulkan loader' library failed to load"),
            EntryLoaderError::EntryLoad(_) => write!(f, "The entry loader failed to initialize"),
        }
    }
}

impl Error for EntryLoaderError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            EntryLoaderError::Library(err) => Some(err),
            EntryLoaderError::EntryLoad(err) => Some(err),
        }
    }
}

impl EntryLoaderImpl {
    /// Load functions using [`libloading`](https://crates.io/crates/libloading).
    ///
    /// Enabled using the `loading` cargo feature.
    ///
    /// For more advanced use cases, take a look at
    /// [`EntryLoader::with_lib_path`] and [`EntryLoader::with_library`].
    pub fn new() -> Result<EntryLoader, EntryLoaderError> {
        EntryLoaderImpl::with_lib_path(LIB_PATH)
    }

    /// Load functions using [`libloading`](https://crates.io/crates/libloading)
    /// providing a custom library path.
    ///
    /// Enabled using the `loading` cargo feature.
    pub fn with_lib_path<P: AsRef<OsStr>>(lib_path: P) -> Result<EntryLoader, EntryLoaderError> {
        let library = unsafe {
            Library::new(lib_path).map_err(|err| EntryLoaderError::Library(LibraryError(err)))?
        };

        let symbol = |library: &mut Library, name| unsafe {
            let cstr = CStr::from_ptr(name);
            let bytes = cstr.to_bytes_with_nul();
            library.get(bytes).ok().map(|symbol| *symbol)
        };

        unsafe {
            CustomEntryLoader::with_library(library, symbol).map_err(EntryLoaderError::EntryLoad)
        }
    }
}
