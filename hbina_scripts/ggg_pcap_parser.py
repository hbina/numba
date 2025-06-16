import numpy as np
import numba

PcapHeader_dtype = np.dtype(
    [
        ("magic_number", "u4"),  # Magic number (4 bytes)
        ("version_major", "u2"),  # Major version (2 bytes)
        ("version_minor", "u2"),  # Minor version (2 bytes)
        ("thiszone", "i4"),  # Time zone offset (4 bytes)
        ("sigfigs", "u4"),  # Timestamp accuracy (4 bytes)
        ("snaplen", "u4"),  # Maximum capture length (4 bytes)
        ("network", "u4"),  # Data link type (4 bytes)
    ]
)
PcapHeader_itemsize = PcapHeader_dtype.itemsize

PcapBlockHeader_dtype = np.dtype(
    [
        ("ts_sec", "u4"),  # Timestamp seconds (4 bytes)
        ("ts_usec", "u4"),  # Timestamp microseconds (4 bytes)
        ("incl_len", "u4"),  # Captured length (4 bytes)
        ("orig_len", "u4"),  # Original length (4 bytes)
    ]
)
PcapBlockHeader_itemsize = PcapBlockHeader_dtype.itemsize


@numba.njit(cache=False)
def numba_main(total_buffer: np.ndarray):
    offset_start = 0
    offset_end = PcapHeader_itemsize
    offset_buffer = total_buffer[offset_start:offset_end]

    pcap_hdr = np.frombuffer(offset_buffer, dtype=PcapHeader_dtype)

    print(pcap_hdr)

    while True:
        offset_start = offset_end
        offset_end = offset_start + PcapBlockHeader_itemsize
        offset_buffer = total_buffer[offset_start:offset_end]

        if len(offset_buffer) != PcapBlockHeader_itemsize:
            break

        pcap_block_hdr = np.frombuffer(offset_buffer, dtype=PcapBlockHeader_dtype)

        print(pcap_block_hdr)

        offset_start = offset_end
        offset_end = offset_start + pcap_block_hdr[0]["incl_len"]
        offset_buffer = total_buffer[offset_start:offset_end]

        if len(offset_buffer) != pcap_block_hdr[0]["incl_len"]:
            break

        pcap_block_data = np.frombuffer(offset_buffer, dtype=np.uint8)

        print(pcap_block_data)

    return


def main():
    total_buffer = np.memmap("/home/hbina085/Downloads/ipv4frags.pcap", dtype=np.uint8)
    print("compiling numba_main")
    numba_main(total_buffer)


if __name__ == "__main__":
    main()
