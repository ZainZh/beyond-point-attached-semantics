import sys

from process_data_semantic_pointwise import main


def has_output_suffix(argv):
    return any(arg == "--output_suffix" or arg.startswith("--output_suffix=") for arg in argv)


def build_hybrid_argv(argv):
    forwarded = list(argv)
    if not has_output_suffix(forwarded):
        forwarded.append("--output_suffix=-objpc-semantic-pointwise-hybrid-semdebugref")
    forwarded.append("--keep_feature_placeholders_in_context")
    return forwarded


if __name__ == "__main__":
    main(build_hybrid_argv(sys.argv[1:]))
