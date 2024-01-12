
import argparse

def load_text_file(text_file_path: str, format: str = 'utf-8') -> str:
    with open(text_file_path, 'r', encoding=format) as source_file:
        content = source_file.read()
    return content

def write_text_file(destination_file: str, text_file_content: str, format: str = 'utf-8') :
    with open(destination_file, 'w', encoding=format) as dest_file:
        dest_file.write(text_file_content)

def get_kernel_code(file_str: str, prefix: str, postfix: str) -> str:
    start_position = file_str.find(prefix) + len(prefix)
    end_position = file_str.find(postfix)
    result = file_str[start_position : end_position]
    return result

def make_cpp_file(pattern_file: str, kernel_code: str, source_cpp_pattern: str) -> str:
    result = pattern_file.replace(source_cpp_pattern, kernel_code)
    return result
    
def prepare_parser():
    parser = argparse.ArgumentParser(description="Kernels C to C++ replacer")
    
    parser.add_argument("--source_cpp_pattern_file", help="path to the source C++ file that will be used as pattern")
    parser.add_argument("--source_cpp_pattern", help="pattern in the source C++ file to be replaced by kernel code")
    parser.add_argument("--source_c_kernel_file", help="path to the source C file with the kernel code to be used while replacing pattern in the source C++ file")
    parser.add_argument("--source_c_kernel_prefix", help="prefix for C kernel code. Used to copy C kernel code to the destination C++ file")
    parser.add_argument("--source_c_kernel_postfix", help="postfix for C kernel code. Used to copy C kernel code to the destination C++ file")
    parser.add_argument("--destination_cpp_file", help="file to the destination C++ file")
    
    return parser

if __name__ == "__main__":
    
    try:
        parser = prepare_parser()
        args = parser.parse_args()
        
        if not all ([args.source_cpp_pattern_file, args.source_cpp_pattern, args.source_c_kernel_file, args.source_c_kernel_prefix, args.source_c_kernel_postfix, args.destination_cpp_file]):
            parser.print_help()
        else:
            print(f"- Load kernel code file {args.source_c_kernel_file}")
            kernel_file = load_text_file(args.source_c_kernel_file)
            
            print(f"- Extract kernel code inside tags {args.source_c_kernel_prefix} .. {args.source_c_kernel_postfix}")
            kernel_code = get_kernel_code(kernel_file, args.source_c_kernel_prefix, args.source_c_kernel_postfix)
            
            print(f"- Load C++ patter file {args.source_cpp_pattern_file}")
            pattern_file = load_text_file(args.source_cpp_pattern_file)
            
            print(f"- Make destination file")
            destination_file = make_cpp_file(pattern_file, kernel_code, args.source_cpp_pattern)

            print(f"- Save destination C++ file {args.destination_cpp_file}")
            write_text_file(args.destination_cpp_file, destination_file)
            
    except Exception as err:
        print(f"*** Error occured: {err}")
        