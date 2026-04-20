"""
Parser for PKS domain annotations from fragments_for_prediction_COREONLY.csv.

The CSV contains a 'fragment_domain_annotations' column with JSON-formatted
domain boundaries:

[
  {"type": "KS", "start": 1, "stop": 429},
  {"type": "KSATL", "start": 430, "stop": 535},
  {"type": "AT", "start": 536, "stop": 830},
  ...
]

Domain naming convention:
- Core domains: KS, AT, DH, ER, KR, ACP, oMT, C, A, PCP, E
- Linkers: Any domain ending in 'L' (e.g., KSATL, ATDHL, ATKRL)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
import re


# Core catalytic domains (non-linker domains)
CORE_DOMAINS = {
    # PKS domains
    'KS',   # Ketosynthase
    'AT',   # Acyltransferase
    'DH',   # Dehydratase
    'ER',   # Enoylreductase
    'KR',   # Ketoreductase
    'ACP',  # Acyl carrier protein
    'oMT',  # O-methyltransferase
    'TE',   # Thioesterase
    # NRPS domains
    'C',    # Condensation
    'A',    # Adenylation
    'PCP',  # Peptidyl carrier protein
    'E',    # Epimerization
}

# Linker pattern: ends with 'L' but isn't a core domain
LINKER_PATTERN = re.compile(r'^[A-Za-z]+L$')


@dataclass
class DomainAnnotation:
    """Single domain annotation."""
    domain_type: str
    start: int  # 1-indexed, inclusive
    stop: int   # 1-indexed, inclusive
    
    @property
    def is_linker(self) -> bool:
        """Check if this is a linker region."""
        return (
            self.domain_type not in CORE_DOMAINS and 
            LINKER_PATTERN.match(self.domain_type) is not None
        )
    
    @property
    def is_core_domain(self) -> bool:
        """Check if this is a core catalytic domain."""
        return self.domain_type in CORE_DOMAINS
    
    @property
    def length(self) -> int:
        return self.stop - self.start + 1
    
    def get_residue_indices(self, zero_indexed: bool = True) -> np.ndarray:
        """Get array of residue indices covered by this domain."""
        if zero_indexed:
            return np.arange(self.start - 1, self.stop)
        return np.arange(self.start, self.stop + 1)


@dataclass
class ModuleAnnotation:
    """Complete annotation for a PKS module."""
    fragment_id: str
    parent_gene_id: str
    fragment_type: str
    fragment_composition: str
    fragment_sequence: str
    domains: List[DomainAnnotation]
    
    @property
    def length(self) -> int:
        return len(self.fragment_sequence)
    
    @property
    def core_domains(self) -> List[DomainAnnotation]:
        """Get only core (non-linker) domains."""
        return [d for d in self.domains if d.is_core_domain]
    
    @property
    def linker_domains(self) -> List[DomainAnnotation]:
        """Get only linker domains."""
        return [d for d in self.domains if d.is_linker]
    
    @property
    def domain_types(self) -> List[str]:
        """Get list of domain types in order."""
        return [d.domain_type for d in self.domains]
    
    @property
    def core_domain_types(self) -> List[str]:
        """Get list of core domain types in order."""
        return [d.domain_type for d in self.core_domains]
    
    def get_domain(self, domain_type: str) -> Optional[DomainAnnotation]:
        """Get annotation for a specific domain type."""
        for domain in self.domains:
            if domain.domain_type == domain_type:
                return domain
        return None
    
    def get_domains(self, domain_types: List[str]) -> List[DomainAnnotation]:
        """Get annotations for multiple domain types."""
        return [d for d in self.domains if d.domain_type in domain_types]
    
    def get_domain_mask(
        self, 
        include_domains: Optional[List[str]] = None,
        exclude_linkers: bool = False
    ) -> np.ndarray:
        """
        Create a boolean mask for specific domains.
        
        Args:
            include_domains: List of domain types to include. If None, include all.
            exclude_linkers: If True, exclude linker regions.
            
        Returns:
            Boolean array of shape (length,)
        """
        mask = np.zeros(self.length, dtype=bool)
        
        for domain in self.domains:
            # Skip linkers if requested
            if exclude_linkers and domain.is_linker:
                continue
            
            # Skip if not in include list
            if include_domains is not None and domain.domain_type not in include_domains:
                continue
            
            indices = domain.get_residue_indices(zero_indexed=True)
            mask[indices] = True
        
        return mask
    
    def get_interface_residues(
        self, 
        interface_width: int = 10
    ) -> np.ndarray:
        """
        Get residue indices at domain interfaces.
        
        Args:
            interface_width: Number of residues on each side of boundary
            
        Returns:
            Boolean array marking interface residues
        """
        mask = np.zeros(self.length, dtype=bool)
        
        for i in range(len(self.domains) - 1):
            boundary = self.domains[i].stop - 1  # 0-indexed boundary
            
            # Mark residues on both sides of boundary
            start = max(0, boundary - interface_width + 1)
            stop = min(self.length, boundary + interface_width + 1)
            mask[start:stop] = True
        
        return mask


class AnnotationParser:
    """
    Parser for PKS module annotations from CSV file.
    
    Expected CSV columns:
    - fragment_id: Unique identifier for the module
    - parent_gene_id: Parent gene identifier
    - fragment_type: Type of fragment (e.g., SingleModuleCore)
    - fragment_composition: Domain composition string
    - fragment_domain_annotations: JSON string with domain boundaries
    - fragment_sequence: Amino acid sequence
    """
    
    def __init__(self, csv_path: Optional[Path] = None):
        """
        Initialize the parser.
        
        Args:
            csv_path: Path to the annotations CSV file
        """
        self.csv_path = Path(csv_path) if csv_path else None
        self._annotations: Dict[str, ModuleAnnotation] = {}
        self._loaded = False
        
        if self.csv_path:
            self.load(self.csv_path)
    
    def load(self, csv_path: Path) -> None:
        """
        Load annotations from CSV file.
        
        Args:
            csv_path: Path to CSV file
        """
        self.csv_path = Path(csv_path)
        
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {csv_path}")
        
        df = pd.read_csv(self.csv_path)
        
        required_columns = [
            'fragment_id', 'parent_gene_id', 'fragment_type',
            'fragment_composition', 'fragment_domain_annotations', 'fragment_sequence'
        ]
        
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        self._annotations = {}
        
        for _, row in df.iterrows():
            try:
                annotation = self._parse_row(row)
                self._annotations[annotation.fragment_id] = annotation
            except Exception as e:
                print(f"Warning: Failed to parse {row.get('fragment_id', 'unknown')}: {e}")
                continue
        
        self._loaded = True
        print(f"Loaded {len(self._annotations)} module annotations")
    
    def _parse_row(self, row: pd.Series) -> ModuleAnnotation:
        """Parse a single row from the CSV."""
        # Parse JSON domain annotations
        domain_json = row['fragment_domain_annotations']
        
        # Handle potential formatting issues
        if isinstance(domain_json, str):
            domain_list = json.loads(domain_json)
        else:
            domain_list = domain_json
        
        domains = []
        for d in domain_list:
            domains.append(DomainAnnotation(
                domain_type=d['type'],
                start=int(d['start']),
                stop=int(d['stop'])
            ))
        
        return ModuleAnnotation(
            fragment_id=row['fragment_id'],
            parent_gene_id=row['parent_gene_id'],
            fragment_type=row['fragment_type'],
            fragment_composition=row['fragment_composition'],
            fragment_sequence=row['fragment_sequence'],
            domains=domains
        )
    
    def get(self, fragment_id: str) -> Optional[ModuleAnnotation]:
        """Get annotation for a specific fragment."""
        return self._annotations.get(fragment_id)
    
    def __getitem__(self, fragment_id: str) -> ModuleAnnotation:
        """Get annotation for a specific fragment."""
        if fragment_id not in self._annotations:
            raise KeyError(f"Fragment not found: {fragment_id}")
        return self._annotations[fragment_id]
    
    def __contains__(self, fragment_id: str) -> bool:
        return fragment_id in self._annotations
    
    def __len__(self) -> int:
        return len(self._annotations)
    
    def __iter__(self):
        return iter(self._annotations.values())
    
    @property
    def fragment_ids(self) -> List[str]:
        """Get all fragment IDs."""
        return list(self._annotations.keys())
    
    def get_by_composition(self, composition: str) -> List[ModuleAnnotation]:
        """Get all modules with a specific domain composition."""
        return [a for a in self._annotations.values() if a.fragment_composition == composition]
    
    def get_by_domain(self, domain_type: str) -> List[ModuleAnnotation]:
        """Get all modules containing a specific domain type."""
        return [a for a in self._annotations.values() if domain_type in a.domain_types]
    
    def get_composition_counts(self) -> Dict[str, int]:
        """Get counts of each domain composition."""
        from collections import Counter
        return dict(Counter(a.fragment_composition for a in self._annotations.values()))
    
    def get_domain_type_counts(self) -> Dict[str, int]:
        """Get counts of each domain type across all modules."""
        from collections import Counter
        counts = Counter()
        for annotation in self._annotations.values():
            counts.update(annotation.domain_types)
        return dict(counts)
    
    def get_unique_sequences(self) -> Dict[str, List[str]]:
        """
        Group fragment_ids by unique sequences.
        
        Returns:
            Dict mapping sequence -> list of fragment_ids with that sequence
        """
        from collections import defaultdict
        seq_to_ids = defaultdict(list)
        for fid, annotation in self._annotations.items():
            seq_to_ids[annotation.fragment_sequence].append(fid)
        return dict(seq_to_ids)


def match_cif_to_annotation(
    cif_filename: str,
    annotation_parser: AnnotationParser
) -> Optional[str]:
    """
    Match a structure filename to its corresponding annotation fragment_id.
    
    This function handles the mapping between structure filenames (``.cif``,
    ``.pdb``, etc.) and the fragment_id format in the annotations CSV.
    
    Args:
        cif_filename: Name of the structure file (without path), e.g. ``foo.pdb``
        annotation_parser: Loaded AnnotationParser
        
    Returns:
        Matching fragment_id or None if not found
    """
    # Remove file extension
    base_name = Path(cif_filename).stem
    
    # Try direct match first
    if base_name in annotation_parser:
        return base_name
    
    # Try common patterns:
    # - CIF might have model number suffix (e.g., _model_0)
    # - CIF might have different separators
    
    # Remove model number suffix if present
    model_pattern = re.compile(r'_model_\d+$')
    cleaned = model_pattern.sub('', base_name)
    
    if cleaned in annotation_parser:
        return cleaned
    
    # Remove fold_ prefix (e.g., fold_bgc0000001p1_abyb1_singlemodule_m0_core)
    if cleaned.lower().startswith('fold_'):
        cleaned = cleaned[5:]
    
    if cleaned in annotation_parser:
        return cleaned
    
    # Try case-insensitive match (CIF may be lowercase, annotations Title-Case)
    cleaned_lower = cleaned.lower()
    for fid in annotation_parser.fragment_ids:
        if fid.lower() == cleaned_lower:
            return fid
    
    # Try replacing underscores with hyphens (singlemodule_m0_core -> SingleModule-M0-Core)
    for fid in annotation_parser.fragment_ids:
        if fid.lower().replace('-', '_') == cleaned_lower:
            return fid
    
    # Try replacing underscores with other separators on base_name
    for sep in ['-', '.']:
        alt_name = base_name.replace('_', sep)
        if alt_name in annotation_parser:
            return alt_name
    
    return None


if __name__ == "__main__":
    # Test the parser
    import sys
    
    if len(sys.argv) > 1:
        parser = AnnotationParser(sys.argv[1])
        
        print(f"\nLoaded {len(parser)} annotations")
        print(f"\nTop 10 domain compositions:")
        for comp, count in sorted(
            parser.get_composition_counts().items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]:
            print(f"  {comp}: {count}")
        
        print(f"\nDomain type counts:")
        for dtype, count in sorted(
            parser.get_domain_type_counts().items(),
            key=lambda x: x[1],
            reverse=True
        )[:15]:
            is_linker = "L" if dtype not in CORE_DOMAINS and LINKER_PATTERN.match(dtype) else ""
            print(f"  {dtype}: {count} {is_linker}")
