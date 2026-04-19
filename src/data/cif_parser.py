"""
CIF file parser for AlphaFold3 PKS module structures.

Extracts:
- Backbone coordinates (N, CA, C, O)
- Sequence
- pLDDT confidence scores (stored in B-factor column)
- Chain information for dimers
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import warnings

try:
    from Bio.PDB import MMCIFParser, PDBParser
    from Bio.PDB.Structure import Structure
    from Bio.PDB.Chain import Chain
    from Bio.PDB.Residue import Residue
except ImportError:
    raise ImportError("BioPython is required. Install with: pip install biopython")


# Standard amino acid mapping
AA_3TO1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    'UNK': 'X', 'MSE': 'M',  # Selenomethionine -> Methionine
}

BACKBONE_ATOMS = ['N', 'CA', 'C', 'O']


@dataclass
class ParsedStructure:
    """Container for parsed structure data."""
    name: str
    sequence: str
    coords: np.ndarray  # Shape: (L, 4, 3) for N, CA, C, O
    plddt: np.ndarray   # Shape: (L,) pLDDT per residue
    chain_ids: np.ndarray  # Shape: (L,) chain ID per residue (0 or 1 for dimer)
    residue_indices: np.ndarray  # Shape: (L,) original residue indices
    
    @property
    def length(self) -> int:
        return len(self.sequence)
    
    @property
    def is_dimer(self) -> bool:
        return len(np.unique(self.chain_ids)) > 1
    
    def get_chain(self, chain_id: int) -> 'ParsedStructure':
        """Extract a single chain from the structure."""
        mask = self.chain_ids == chain_id
        return ParsedStructure(
            name=f"{self.name}_chain{chain_id}",
            sequence=''.join([self.sequence[i] for i in np.where(mask)[0]]),
            coords=self.coords[mask],
            plddt=self.plddt[mask],
            chain_ids=self.chain_ids[mask],
            residue_indices=self.residue_indices[mask]
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format compatible with ProteinMPNN."""
        return {
            'name': self.name,
            'seq': self.sequence,
            'coords': {
                'N': self.coords[:, 0, :].tolist(),
                'CA': self.coords[:, 1, :].tolist(),
                'C': self.coords[:, 2, :].tolist(),
                'O': self.coords[:, 3, :].tolist(),
            },
            'plddt': self.plddt.tolist(),
            'chain_ids': self.chain_ids.tolist(),
        }


class CIFParser:
    """
    Parser for CIF/mmCIF files from AlphaFold3 predictions.
    
    AlphaFold stores pLDDT confidence scores in the B-factor column.
    These scores range from 0-100:
    - >90: Very high confidence
    - 70-90: Confident
    - 50-70: Low confidence
    - <50: Very low confidence (often disordered)
    """
    
    def __init__(self, quiet: bool = True):
        """
        Initialize the parser.
        
        Args:
            quiet: Suppress BioPython warnings
        """
        self.mmcif_parser = MMCIFParser(QUIET=quiet)
        self.pdb_parser = PDBParser(QUIET=quiet)
        
    def parse(self, filepath: Union[str, Path]) -> ParsedStructure:
        """
        Parse a CIF or PDB file.
        
        Args:
            filepath: Path to structure file
            
        Returns:
            ParsedStructure with coordinates, sequence, and pLDDT
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Structure file not found: {filepath}")
        
        # Choose parser based on file extension
        if filepath.suffix.lower() in ['.cif', '.mmcif']:
            structure = self.mmcif_parser.get_structure(filepath.stem, str(filepath))
        elif filepath.suffix.lower() in ['.pdb', '.ent']:
            structure = self.pdb_parser.get_structure(filepath.stem, str(filepath))
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        return self._extract_data(structure, filepath.stem)
    
    def _extract_data(self, structure: Structure, name: str) -> ParsedStructure:
        """Extract sequence, coordinates, and pLDDT from BioPython structure."""
        
        sequences = []
        coords_list = []
        plddt_list = []
        chain_ids = []
        residue_indices = []
        
        # Process each model (usually just one for AF predictions)
        model = structure[0]
        
        for chain_idx, chain in enumerate(model.get_chains()):
            for residue in chain.get_residues():
                # Skip heteroatoms and water
                if residue.id[0] != ' ':
                    continue
                
                # Get residue name and convert to single letter
                resname = residue.resname.strip()
                aa = AA_3TO1.get(resname, 'X')
                
                # Extract backbone coordinates
                backbone_coords = np.zeros((4, 3), dtype=np.float32)
                backbone_coords[:] = np.nan
                
                plddt_values = []
                
                for atom_idx, atom_name in enumerate(BACKBONE_ATOMS):
                    if atom_name in residue:
                        atom = residue[atom_name]
                        backbone_coords[atom_idx] = atom.get_coord()
                        plddt_values.append(atom.get_bfactor())
                
                # Skip residues with missing backbone atoms
                if np.isnan(backbone_coords).any():
                    warnings.warn(f"Missing backbone atoms in {name} residue {residue.id}")
                    continue
                
                # pLDDT is typically the same for all atoms in a residue
                # Use CA if available, otherwise average
                plddt = np.mean(plddt_values) if plddt_values else 0.0
                
                sequences.append(aa)
                coords_list.append(backbone_coords)
                plddt_list.append(plddt)
                chain_ids.append(chain_idx)
                residue_indices.append(residue.id[1])
        
        if not sequences:
            raise ValueError(f"No valid residues found in {name}")
        
        return ParsedStructure(
            name=name,
            sequence=''.join(sequences),
            coords=np.array(coords_list, dtype=np.float32),
            plddt=np.array(plddt_list, dtype=np.float32),
            chain_ids=np.array(chain_ids, dtype=np.int32),
            residue_indices=np.array(residue_indices, dtype=np.int32)
        )
    
    def parse_batch(
        self, 
        filepaths: List[Union[str, Path]], 
        n_workers: int = 1,
        show_progress: bool = True
    ) -> List[ParsedStructure]:
        """
        Parse multiple structure files.
        
        Args:
            filepaths: List of paths to structure files
            n_workers: Number of parallel workers (1 = sequential)
            show_progress: Show progress bar
            
        Returns:
            List of ParsedStructure objects
        """
        from tqdm import tqdm
        
        structures = []
        iterator = tqdm(filepaths, desc="Parsing structures") if show_progress else filepaths
        
        for filepath in iterator:
            try:
                structure = self.parse(filepath)
                structures.append(structure)
            except Exception as e:
                warnings.warn(f"Failed to parse {filepath}: {e}")
                continue
        
        return structures


def compute_ca_distances(coords: np.ndarray) -> np.ndarray:
    """
    Compute pairwise CA distances.
    
    Args:
        coords: Shape (L, 4, 3) backbone coordinates
        
    Returns:
        Shape (L, L) distance matrix
    """
    ca_coords = coords[:, 1, :]  # CA is index 1
    diff = ca_coords[:, np.newaxis, :] - ca_coords[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=-1))
    return distances


def build_knn_graph(coords: np.ndarray, k: int = 48) -> np.ndarray:
    """
    Build K-nearest neighbor graph based on CA distances.
    
    This matches ProteinMPNN's default of K=48 neighbors.
    
    Args:
        coords: Shape (L, 4, 3) backbone coordinates
        k: Number of neighbors
        
    Returns:
        Shape (L, K) indices of K nearest neighbors for each residue
    """
    distances = compute_ca_distances(coords)
    
    # Set diagonal to inf to exclude self
    np.fill_diagonal(distances, np.inf)
    
    # Get K nearest neighbors
    k = min(k, len(coords) - 1)
    knn_indices = np.argsort(distances, axis=1)[:, :k]
    
    return knn_indices


if __name__ == "__main__":
    # Test the parser
    import sys
    
    if len(sys.argv) > 1:
        parser = CIFParser()
        structure = parser.parse(sys.argv[1])
        print(f"Parsed: {structure.name}")
        print(f"  Length: {structure.length}")
        print(f"  Sequence: {structure.sequence[:50]}...")
        print(f"  pLDDT range: {structure.plddt.min():.1f} - {structure.plddt.max():.1f}")
        print(f"  Is dimer: {structure.is_dimer}")
