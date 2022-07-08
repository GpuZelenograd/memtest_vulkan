extern crate proc_macro;
use proc_macro::TokenStream;


struct InlineSpirV(TokenStream);

fn naga_compile(
    src: &str,
) -> Result<Vec<u32>, String> {
    use naga::{
        valid::{ValidationFlags, Validator, Capabilities},
        ShaderStage,
        front::glsl::{Parser, Options}
    };

    let module = Parser::default().parse(&Options::from(ShaderStage::Compute), src);
    let module = module.map_err(|errs| format!("{:?}", errs))?;
    let mut opts = naga::back::spv::Options::default();
    opts.lang_version = (1, 0);
    opts.flags = naga::back::spv::WriterFlags::DEBUG;

    // Attempt to validate WGSL, error if invalid
    let info = Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .map_err(|e| format!("{:?}", e))?;
    let spv = naga::back::spv::write_vec(&module, &info, &opts, None)
        .map_err(|e| format!("{:?}", e))?;
    Ok(spv)
}

fn gen_token_stream(feedback: Vec<u32>) -> TokenStream {
    (quote::quote! {
        {
            &[#(#feedback),*]
        }
    }).into()
}

impl syn::parse::Parse for InlineSpirV {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let src = input.parse::<syn::LitStr>().map(|x| x.value())?;
        let compiled = naga_compile(&src)
            .map_err(|e| syn::parse::Error::new(input.span(), e))?;
        Ok(Self(gen_token_stream(compiled)))
    }
}


/// Compile inline shader source and embed the SPIR-V binary word sequence.
/// Returns a `&'static [u32]`.
#[proc_macro]
pub fn compiled_vk_compute_spirv(tokens: TokenStream) -> TokenStream {
    syn::parse_macro_input!(tokens as InlineSpirV).0
}

