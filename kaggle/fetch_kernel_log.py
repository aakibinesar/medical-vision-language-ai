"""Windows workaround for the Kaggle SDK's kernels_output() cp1252 crash:
`kernels_output()` downloads a completed kernel's log by opening the local
file with the OS default encoding (cp1252 on Windows) instead of UTF-8,
which crashes if the log contains non-Latin1 characters (pip's own
progress-bar output, for instance). This calls the same underlying API
directly and writes the log with UTF-8 itself, bypassing the buggy write.

Usage: python fetch_kernel_log.py <owner/kernel-slug> <output-path>
"""
import sys

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest


def main():
    kernel, outpath = sys.argv[1], sys.argv[2]

    api = KaggleApi()
    api.authenticate()
    owner_slug, kernel_slug, _version = api.parse_kernel_string(kernel)

    with api.build_kaggle_client() as kaggle:
        request = ApiListKernelSessionOutputRequest()
        request.user_name = owner_slug
        request.kernel_slug = kernel_slug
        response = kaggle.kernels.kernels_api_client.list_kernel_session_output(request)

    log = response.log or ""
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(log)
    print(f"Wrote {len(log)} chars to {outpath}")


if __name__ == "__main__":
    main()
