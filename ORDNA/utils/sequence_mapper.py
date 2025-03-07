from Bio.Data import IUPACData
from typing import Optional, List
import math

class SequenceMapper():
    def __init__(self) -> None:
        iupac_letters = IUPACData.ambiguous_dna_letters.lower()
        iupac_letters_indices = list(range(1, len(iupac_letters)+1))

        self.iupac_dict = dict(zip(iupac_letters, iupac_letters_indices))
        self.iupac_dict["pad"] = 0
        self.iupac_dict["u"] = len(iupac_letters) + 1  # u is missing in IUPACData for some reason

    def map_seq(self, seq: str, pad_to_len: Optional[int] = None) -> List[int]:
        # Gestisci il caso in cui seq non sia una stringa (ad esempio, sia NA o altro)
        if not isinstance(seq, str):
            # Se è un float e risulta NaN, lo sostituisco con una stringa vuota
            if isinstance(seq, float) and math.isnan(seq):
                seq = ""
            else:
                seq = str(seq)
                
        # Mappa ogni nucleotide in minuscolo
        mapped_seq = [self.iupac_dict[nucleotide.lower()] for nucleotide in seq if nucleotide]
        if pad_to_len is None:
            return mapped_seq
        elif pad_to_len - len(mapped_seq) > 0:
            mapped_seq = mapped_seq + [self.iupac_dict["pad"]] * (pad_to_len - len(mapped_seq))
        elif pad_to_len - len(mapped_seq) < 0:
            raise Exception("--sequence_length needs to be set to an upper bound for all initial input sequence lengths.")
        return mapped_seq

    def map_seq_list(self, seq_list: List[str], pad_to_len: Optional[int] = None) -> List[List[int]]:
        return [self.map_seq(seq, pad_to_len) for seq in seq_list]
