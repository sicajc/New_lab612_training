import heapq
import os

"""
author: Bhrigu Srivastava
website: https:bhrigu.me
"""


class HuffmanCoding:
    def __init__(self, path):
        self.path = path  # The input file path
        self.heap = []  # The empty list for minHeap
        self.codes = {}  # The dictionary for encoded binary and key
        self.reverse_mapping = {}  # Helps mapping the codes back to character

    class HeapNode:  # The node of the huffman tree
        def __init__(self, char, freq):
            self.char = char  # Charactor of the node
            self.freq = freq  # Frequency of the node
            self.left = None  # Left child
            self.right = None  # Right child

        # defining comparators less_than and equals
        def __lt__(self, other):
            return self.freq < other.freq

        def __eq__(self, other):
            if(other == None):
                return False
            if(not isinstance(other, HeapNode)):
                return False
            return self.freq == other.freq

    # functions for compression:
    def make_frequency_dict(self, text):
        frequency = {}  # Create a dictionary for traversal
        for character in text:
            if not character in frequency:
                frequency[character] = 0
            frequency[character] += 1
        return frequency

    def make_heap(self, frequency):
        for key in frequency:
            # Creating min heap by traversing each key after finding out the frequency
            node = self.HeapNode(key, frequency[key])
            # Use heapq function to insert node and then heapify
            heapq.heappush(self.heap, node)

    def merge_nodes(self):
        # Building huffman tree
        HeapIsNotEmpty = len(self.heap) > 1
        while(HeapIsNotEmpty):
            # Get 2 nodes from the root of heap
            node1 = heapq.heappop(self.heap)
            node2 = heapq.heappop(self.heap)

            # Merge them together to create a new node which becomes the parent of thses two nodes
            merged = self.HeapNode(None, node1.freq + node2.freq)

            # Connect the original node with the right and left node you just extracted
            merged.left = node1
            merged.right = node2

            # Push the newly generated node back into heap
            heapq.heappush(self.heap, merged)

    def make_codes_helper(self, root, current_code):
        # Recursive function that helps us generate encoding for the heap we just created
        if root is None:
            return

        if(root.char != None):
            self.codes[root.char] = current_code
            # Mapping from code to character
            self.reverse_mapping[current_code] = root.char
            return

        # Recursive call to perform the same task on right and left nodes to generate 01 binary code for each char
        self.make_codes_helper(root.left, current_code + "0")
        self.make_codes_helper(root.right, current_code + "1")

    def make_codes(self):
        root = heapq.heappop(self.heap)
        current_code = ""
        # The recursive function to perform bit assiangment for each node
        self.make_codes_helper(root, current_code)

    def get_encoded_text(self, text):
        encoded_text = ""
        for character in text:
            encoded_text += self.codes[character]
        return encoded_text

    def pad_encoded_text(self, encoded_text):
        # Pad the encoded text to 8 bit, since we useful transfer data in 8 bits.
        extra_padding = 8 - len(encoded_text) % 8
        for i in range(extra_padding):
            encoded_text += "0"

        # Since extra_padding is an integer, we want to format it into byte code
        padded_info = "{0:08b}".format(extra_padding)
        encoded_text = padded_info + encoded_text
        return encoded_text

    def get_byte_array(self, padded_encoded_text):
        # Convert bit string into byte array
        if(len(padded_encoded_text) % 8 != 0):
            print("Encoded text not padded properly")
            exit(0)

        b = bytearray()
        for i in range(0, len(padded_encoded_text), 8):
            byte = padded_encoded_text[i:i+8]
            b.append(int(byte, 2))
        return b

    # Notice that this kind of coding style cleverly captures the steps of each algorithms
    # Great Coding style that must be learnt
    def compress(self):
        filename, file_extension = os.path.splitext(
            self.path)  # First read in the file
        output_path = filename + ".bin"  # The output path for the result

        with open(self.path, 'r+') as file, open(output_path, 'wb') as output:
            text = file.read()
            text = text.rstrip()

            # Function for frequency dictionary generation
            frequency = self.make_frequency_dict(text)
            self.make_heap(frequency)  # Create MinHeap
            self.merge_nodes()  # Merging
            self.make_codes()

            encoded_text = self.get_encoded_text(text)
            padded_encoded_text = self.pad_encoded_text(encoded_text)

            b = self.get_byte_array(padded_encoded_text)
            output.write(bytes(b))

        print("Compressed")
        return output_path

    """ functions for decompression: """

    def remove_padding(self, padded_encoded_text):
        padded_info = padded_encoded_text[:8]
        extra_padding = int(padded_info, 2)

        padded_encoded_text = padded_encoded_text[8:]
        encoded_text = padded_encoded_text[:-1*extra_padding]

        return encoded_text

    def decode_text(self, encoded_text):
        current_code = ""
        decoded_text = ""

        for bit in encoded_text:
            current_code += bit
            if(current_code in self.reverse_mapping):
                character = self.reverse_mapping[current_code]
                decoded_text += character
                current_code = ""

        return decoded_text

    def decompress(self, input_path):
        filename, file_extension = os.path.splitext(self.path)
        output_path = filename + "_decompressed" + ".txt"

        with open(input_path, 'rb') as file, open(output_path, 'w') as output:
            bit_string = ""

            byte = file.read(1)
            while(len(byte) > 0):
                byte = ord(byte)
                bits = bin(byte)[2:].rjust(8, '0')
                bit_string += bits
                byte = file.read(1)

            encoded_text = self.remove_padding(bit_string)

            decompressed_text = self.decode_text(encoded_text)

            output.write(decompressed_text)

        print("Decompressed")
        return output_path
