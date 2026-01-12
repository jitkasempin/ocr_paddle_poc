"""
Repetition Detector Module for OCR Output

This module provides robust solutions to detect exact phrase repetition in markdown/text
output from OCR models. It's designed to integrate with invoice data extractor projects
where OCR models may produce repetitive text due to complex invoice images.

Features:
- Multiple detection algorithms (Suffix Array, N-gram, Rolling Hash, Regex)
- Configurable minimum phrase length and repetition thresholds
- Support for both consecutive and non-consecutive repetitions
- Detailed reporting with positions and repetition counts

Author: Generated for Invoice Data Extractor Project
"""

import re
import hashlib
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from enum import Enum


class RepetitionType(Enum):
    """Types of repetition detected."""
    CONSECUTIVE = "consecutive"
    NON_CONSECUTIVE = "non_consecutive"
    MIXED = "mixed"


@dataclass
class RepetitionMatch:
    """Represents a detected repetition in the text."""
    phrase: str
    count: int
    positions: List[int]
    repetition_type: RepetitionType
    phrase_length: int = field(init=False)
    
    def __post_init__(self):
        self.phrase_length = len(self.phrase)
    
    def is_consecutive(self) -> bool:
        """Check if all occurrences are consecutive."""
        if len(self.positions) < 2:
            return False
        for i in range(1, len(self.positions)):
            expected_pos = self.positions[i-1] + self.phrase_length
            # Allow small gap for whitespace/newlines
            if abs(self.positions[i] - expected_pos) > 10:
                return False
        return True


@dataclass
class DetectionResult:
    """Result of repetition detection."""
    has_repetition: bool
    repetitions: List[RepetitionMatch]
    total_repetition_count: int
    repetitive_content_ratio: float
    original_length: int
    detection_method: str
    
    def get_summary(self) -> str:
        """Generate a human-readable summary of the detection results."""
        if not self.has_repetition:
            return "No significant repetitions detected."
        
        summary_lines = [
            f"Repetition Detection Summary:",
            f"  - Total repetitions found: {len(self.repetitions)}",
            f"  - Total repeated occurrences: {self.total_repetition_count}",
            f"  - Repetitive content ratio: {self.repetitive_content_ratio:.2%}",
            f"  - Detection method: {self.detection_method}",
            f"\nTop repetitions:"
        ]
        
        # Show top 5 repetitions by count
        sorted_reps = sorted(self.repetitions, key=lambda x: x.count, reverse=True)[:5]
        for i, rep in enumerate(sorted_reps, 1):
            preview = rep.phrase[:50] + "..." if len(rep.phrase) > 50 else rep.phrase
            preview = preview.replace('\n', '\\n')
            summary_lines.append(f"  {i}. [{rep.count}x] \"{preview}\"")
        
        return "\n".join(summary_lines)


class RepetitionDetector:
    """
    Main class for detecting repetitive text in OCR output.
    
    This detector is optimized for finding exact phrase repetitions that commonly
    occur when OCR models process complex invoice images and produce repeated
    text segments.
    """
    
    def __init__(
        self,
        min_phrase_length: int = 20,
        min_repetitions: int = 2,
        use_combined_detection: bool = True
    ):
        """
        Initialize the repetition detector.
        
        Args:
            min_phrase_length: Minimum length of phrase to consider as repetition
            min_repetitions: Minimum number of occurrences to flag as repetition
            use_combined_detection: Use multiple algorithms for better accuracy
        """
        self.min_phrase_length = min_phrase_length
        self.min_repetitions = min_repetitions
        self.use_combined_detection = use_combined_detection
    
    def detect(self, text: str) -> DetectionResult:
        """
        Detect repetitions in the given text using the best available method.
        
        Args:
            text: The text to analyze for repetitions
            
        Returns:
            DetectionResult containing all detected repetitions
        """
        if not text or len(text) < self.min_phrase_length * 2:
            return DetectionResult(
                has_repetition=False,
                repetitions=[],
                total_repetition_count=0,
                repetitive_content_ratio=0.0,
                original_length=len(text) if text else 0,
                detection_method="none"
            )
        
        if self.use_combined_detection:
            return self._combined_detection(text)
        else:
            return self._suffix_array_detection(text)
    
    def _combined_detection(self, text: str) -> DetectionResult:
        """
        Use multiple detection methods and combine results for better accuracy.
        """
        all_repetitions: Dict[str, RepetitionMatch] = {}
        
        # Method 1: Suffix Array / LRS approach
        suffix_result = self._suffix_array_detection(text)
        for rep in suffix_result.repetitions:
            if rep.phrase not in all_repetitions:
                all_repetitions[rep.phrase] = rep
        
        # Method 2: N-gram analysis
        ngram_result = self._ngram_detection(text)
        for rep in ngram_result.repetitions:
            if rep.phrase not in all_repetitions:
                all_repetitions[rep.phrase] = rep
        
        # Method 3: Rolling hash for longer patterns
        rolling_result = self._rolling_hash_detection(text)
        for rep in rolling_result.repetitions:
            if rep.phrase not in all_repetitions:
                all_repetitions[rep.phrase] = rep
        
        # Filter out substrings of larger repetitions
        filtered_repetitions = self._filter_substrings(list(all_repetitions.values()))
        
        # Calculate metrics
        total_count = sum(rep.count for rep in filtered_repetitions)
        repetitive_chars = sum(rep.phrase_length * (rep.count - 1) for rep in filtered_repetitions)
        ratio = repetitive_chars / len(text) if text else 0.0
        
        return DetectionResult(
            has_repetition=len(filtered_repetitions) > 0,
            repetitions=filtered_repetitions,
            total_repetition_count=total_count,
            repetitive_content_ratio=ratio,
            original_length=len(text),
            detection_method="combined"
        )
    
    def _suffix_array_detection(self, text: str) -> DetectionResult:
        """
        Detect repetitions using suffix array and longest repeated substring approach.
        
        This method builds a suffix array and finds repeated substrings by
        comparing adjacent suffixes in the sorted array.
        """
        n = len(text)
        if n < self.min_phrase_length * 2:
            return self._empty_result(text, "suffix_array")
        
        # Build suffix array (simplified version)
        suffixes = [(text[i:], i) for i in range(n)]
        suffixes.sort(key=lambda x: x[0])
        
        # Find repeated substrings using LCP (Longest Common Prefix)
        repetitions: Dict[str, List[int]] = defaultdict(list)
        
        for i in range(1, len(suffixes)):
            lcp_length = self._lcp(suffixes[i-1][0], suffixes[i][0])
            if lcp_length >= self.min_phrase_length:
                phrase = suffixes[i][0][:lcp_length]
                repetitions[phrase].append(suffixes[i-1][1])
                repetitions[phrase].append(suffixes[i][1])
        
        # Convert to RepetitionMatch objects
        matches = []
        for phrase, positions in repetitions.items():
            unique_positions = sorted(set(positions))
            if len(unique_positions) >= self.min_repetitions:
                rep_type = self._determine_repetition_type(unique_positions, len(phrase))
                matches.append(RepetitionMatch(
                    phrase=phrase,
                    count=len(unique_positions),
                    positions=unique_positions,
                    repetition_type=rep_type
                ))
        
        return self._create_result(text, matches, "suffix_array")
    
    def _ngram_detection(self, text: str) -> DetectionResult:
        """
        Detect repetitions using N-gram frequency analysis.
        
        This method generates character n-grams and identifies those that
        appear multiple times.
        """
        n = self.min_phrase_length
        if len(text) < n * 2:
            return self._empty_result(text, "ngram")
        
        # Generate n-grams with positions
        ngram_positions: Dict[str, List[int]] = defaultdict(list)
        
        for i in range(len(text) - n + 1):
            ngram = text[i:i + n]
            ngram_positions[ngram].append(i)
        
        # Find repeated n-grams and extend them
        repetitions: Dict[str, List[int]] = {}
        
        for ngram, positions in ngram_positions.items():
            if len(positions) >= self.min_repetitions:
                # Try to extend the n-gram
                extended = self._extend_ngram(text, positions, n)
                if extended:
                    phrase, ext_positions = extended
                    if phrase not in repetitions:
                        repetitions[phrase] = ext_positions
        
        # Convert to RepetitionMatch objects
        matches = []
        for phrase, positions in repetitions.items():
            if len(positions) >= self.min_repetitions:
                rep_type = self._determine_repetition_type(positions, len(phrase))
                matches.append(RepetitionMatch(
                    phrase=phrase,
                    count=len(positions),
                    positions=positions,
                    repetition_type=rep_type
                ))
        
        return self._create_result(text, matches, "ngram")
    
    def _rolling_hash_detection(self, text: str) -> DetectionResult:
        """
        Detect repetitions using rolling hash (Rabin-Karp style) approach.
        
        This method uses a rolling hash to efficiently find repeated substrings
        of various lengths.
        """
        if len(text) < self.min_phrase_length * 2:
            return self._empty_result(text, "rolling_hash")
        
        repetitions: Dict[str, List[int]] = {}
        
        # Try different window sizes
        for window_size in [self.min_phrase_length, self.min_phrase_length * 2, 
                           self.min_phrase_length * 3]:
            if window_size > len(text) // 2:
                continue
            
            hash_positions: Dict[int, List[int]] = defaultdict(list)
            
            # Calculate initial hash
            base = 256
            mod = 10**9 + 7
            
            current_hash = 0
            base_power = 1
            
            for i in range(window_size):
                current_hash = (current_hash * base + ord(text[i])) % mod
                if i < window_size - 1:
                    base_power = (base_power * base) % mod
            
            hash_positions[current_hash].append(0)
            
            # Rolling hash
            for i in range(1, len(text) - window_size + 1):
                # Remove leading character and add trailing character
                current_hash = (current_hash - ord(text[i-1]) * base_power) % mod
                current_hash = (current_hash * base + ord(text[i + window_size - 1])) % mod
                current_hash = (current_hash + mod) % mod
                
                hash_positions[current_hash].append(i)
            
            # Verify matches (handle hash collisions)
            for hash_val, positions in hash_positions.items():
                if len(positions) >= self.min_repetitions:
                    # Group by actual content
                    content_groups: Dict[str, List[int]] = defaultdict(list)
                    for pos in positions:
                        content = text[pos:pos + window_size]
                        content_groups[content].append(pos)
                    
                    for phrase, phrase_positions in content_groups.items():
                        if len(phrase_positions) >= self.min_repetitions:
                            if phrase not in repetitions:
                                repetitions[phrase] = phrase_positions
        
        # Convert to RepetitionMatch objects
        matches = []
        for phrase, positions in repetitions.items():
            rep_type = self._determine_repetition_type(positions, len(phrase))
            matches.append(RepetitionMatch(
                phrase=phrase,
                count=len(positions),
                positions=positions,
                repetition_type=rep_type
            ))
        
        return self._create_result(text, matches, "rolling_hash")
    
    def _lcp(self, s1: str, s2: str) -> int:
        """Calculate the longest common prefix length of two strings."""
        min_len = min(len(s1), len(s2))
        for i in range(min_len):
            if s1[i] != s2[i]:
                return i
        return min_len
    
    def _extend_ngram(self, text: str, positions: List[int], n: int) -> Optional[Tuple[str, List[int]]]:
        """
        Try to extend an n-gram to find the longest repeated phrase.
        """
        if len(positions) < 2:
            return None
        
        # Start with the n-gram
        base_phrase = text[positions[0]:positions[0] + n]
        
        # Try to extend forward
        extended_length = n
        while True:
            can_extend = True
            for pos in positions:
                if pos + extended_length >= len(text):
                    can_extend = False
                    break
                if text[pos + extended_length] != text[positions[0] + extended_length]:
                    can_extend = False
                    break
            
            if can_extend:
                extended_length += 1
            else:
                break
        
        if extended_length > n:
            phrase = text[positions[0]:positions[0] + extended_length]
            return (phrase, positions)
        
        return (base_phrase, positions)
    
    def _determine_repetition_type(self, positions: List[int], phrase_length: int) -> RepetitionType:
        """Determine if repetitions are consecutive, non-consecutive, or mixed."""
        if len(positions) < 2:
            return RepetitionType.NON_CONSECUTIVE
        
        consecutive_count = 0
        for i in range(1, len(positions)):
            gap = positions[i] - positions[i-1]
            if gap <= phrase_length + 10:  # Allow small gap for whitespace
                consecutive_count += 1
        
        if consecutive_count == len(positions) - 1:
            return RepetitionType.CONSECUTIVE
        elif consecutive_count == 0:
            return RepetitionType.NON_CONSECUTIVE
        else:
            return RepetitionType.MIXED
    
    def _filter_substrings(self, matches: List[RepetitionMatch]) -> List[RepetitionMatch]:
        """Remove repetitions that are substrings of larger repetitions."""
        if not matches:
            return []
        
        # Sort by phrase length descending
        sorted_matches = sorted(matches, key=lambda x: x.phrase_length, reverse=True)
        
        filtered = []
        for match in sorted_matches:
            is_substring = False
            for existing in filtered:
                if match.phrase in existing.phrase:
                    is_substring = True
                    break
            if not is_substring:
                filtered.append(match)
        
        return filtered
    
    def _empty_result(self, text: str, method: str) -> DetectionResult:
        """Create an empty detection result."""
        return DetectionResult(
            has_repetition=False,
            repetitions=[],
            total_repetition_count=0,
            repetitive_content_ratio=0.0,
            original_length=len(text) if text else 0,
            detection_method=method
        )
    
    def _create_result(self, text: str, matches: List[RepetitionMatch], method: str) -> DetectionResult:
        """Create a detection result from matches."""
        if not matches:
            return self._empty_result(text, method)
        
        total_count = sum(rep.count for rep in matches)
        repetitive_chars = sum(rep.phrase_length * (rep.count - 1) for rep in matches)
        ratio = repetitive_chars / len(text) if text else 0.0
        
        return DetectionResult(
            has_repetition=True,
            repetitions=matches,
            total_repetition_count=total_count,
            repetitive_content_ratio=ratio,
            original_length=len(text),
            detection_method=method
        )


class ConsecutiveRepetitionDetector:
    """
    Specialized detector for consecutive exact phrase repetitions.
    
    This detector is optimized for the specific case where OCR models
    produce the same text segment multiple times in a row.
    """
    
    def __init__(self, min_phrase_length: int = 20, min_repetitions: int = 2):
        """
        Initialize the consecutive repetition detector.
        
        Args:
            min_phrase_length: Minimum length of phrase to consider
            min_repetitions: Minimum number of consecutive occurrences
        """
        self.min_phrase_length = min_phrase_length
        self.min_repetitions = min_repetitions
    
    def detect(self, text: str) -> DetectionResult:
        """
        Detect consecutive repetitions using regex-based approach.
        
        This method is highly efficient for finding exact consecutive repetitions.
        """
        if not text or len(text) < self.min_phrase_length * 2:
            return DetectionResult(
                has_repetition=False,
                repetitions=[],
                total_repetition_count=0,
                repetitive_content_ratio=0.0,
                original_length=len(text) if text else 0,
                detection_method="consecutive_regex"
            )
        
        matches = []
        
        # Pattern to find consecutive repetitions
        # This regex finds any sequence that repeats 2+ times consecutively
        # Using a greedy approach to find longer patterns first
        
        for length in range(min(500, len(text) // 2), self.min_phrase_length - 1, -1):
            # Build pattern for this length
            pattern = r'(.{' + str(length) + r',}?)\1{' + str(self.min_repetitions - 1) + r',}'
            
            for match in re.finditer(pattern, text, re.DOTALL):
                phrase = match.group(1)
                full_match = match.group(0)
                count = len(full_match) // len(phrase)
                
                # Check if this is already covered by a longer match
                is_covered = False
                for existing in matches:
                    if match.start() >= existing.positions[0] and \
                       match.end() <= existing.positions[0] + existing.phrase_length * existing.count:
                        is_covered = True
                        break
                
                if not is_covered and len(phrase) >= self.min_phrase_length:
                    matches.append(RepetitionMatch(
                        phrase=phrase,
                        count=count,
                        positions=[match.start() + i * len(phrase) for i in range(count)],
                        repetition_type=RepetitionType.CONSECUTIVE
                    ))
        
        # Calculate metrics
        total_count = sum(rep.count for rep in matches)
        repetitive_chars = sum(rep.phrase_length * (rep.count - 1) for rep in matches)
        ratio = repetitive_chars / len(text) if text else 0.0
        
        return DetectionResult(
            has_repetition=len(matches) > 0,
            repetitions=matches,
            total_repetition_count=total_count,
            repetitive_content_ratio=ratio,
            original_length=len(text),
            detection_method="consecutive_regex"
        )


class HTMLTableRepetitionDetector:
    """
    Specialized detector for HTML table row repetitions.
    
    This detector is specifically designed for OCR output that contains
    HTML table structures, which is common in invoice processing.
    """
    
    def __init__(self, min_repetitions: int = 2):
        """
        Initialize the HTML table repetition detector.
        
        Args:
            min_repetitions: Minimum number of occurrences to flag
        """
        self.min_repetitions = min_repetitions
    
    def detect(self, text: str) -> DetectionResult:
        """
        Detect repeated HTML table rows in the text.
        """
        if not text:
            return DetectionResult(
                has_repetition=False,
                repetitions=[],
                total_repetition_count=0,
                repetitive_content_ratio=0.0,
                original_length=0,
                detection_method="html_table"
            )
        
        matches = []
        
        # Pattern to match table rows
        row_pattern = r'<tr[^>]*>.*?</tr>'
        
        # Find all table rows
        rows = re.findall(row_pattern, text, re.DOTALL | re.IGNORECASE)
        
        # Count occurrences
        row_counts = Counter(rows)
        
        # Find repeated rows
        for row, count in row_counts.items():
            if count >= self.min_repetitions and len(row) >= 20:
                # Find all positions
                positions = [m.start() for m in re.finditer(re.escape(row), text)]
                
                rep_type = self._determine_type(positions, len(row))
                matches.append(RepetitionMatch(
                    phrase=row,
                    count=count,
                    positions=positions,
                    repetition_type=rep_type
                ))
        
        # Also check for repeated td elements
        td_pattern = r'<td[^>]*>.*?</td>'
        tds = re.findall(td_pattern, text, re.DOTALL | re.IGNORECASE)
        td_counts = Counter(tds)
        
        for td, count in td_counts.items():
            if count >= self.min_repetitions * 2 and len(td) >= 30:  # Higher threshold for td
                positions = [m.start() for m in re.finditer(re.escape(td), text)]
                rep_type = self._determine_type(positions, len(td))
                matches.append(RepetitionMatch(
                    phrase=td,
                    count=count,
                    positions=positions,
                    repetition_type=rep_type
                ))
        
        # Calculate metrics
        total_count = sum(rep.count for rep in matches)
        repetitive_chars = sum(rep.phrase_length * (rep.count - 1) for rep in matches)
        ratio = repetitive_chars / len(text) if text else 0.0
        
        return DetectionResult(
            has_repetition=len(matches) > 0,
            repetitions=matches,
            total_repetition_count=total_count,
            repetitive_content_ratio=ratio,
            original_length=len(text),
            detection_method="html_table"
        )
    
    def _determine_type(self, positions: List[int], phrase_length: int) -> RepetitionType:
        """Determine repetition type based on positions."""
        if len(positions) < 2:
            return RepetitionType.NON_CONSECUTIVE
        
        consecutive_count = 0
        for i in range(1, len(positions)):
            if positions[i] - positions[i-1] <= phrase_length + 50:
                consecutive_count += 1
        
        if consecutive_count == len(positions) - 1:
            return RepetitionType.CONSECUTIVE
        elif consecutive_count == 0:
            return RepetitionType.NON_CONSECUTIVE
        return RepetitionType.MIXED


def detect_repetition(
    text: str,
    min_phrase_length: int = 10,
    min_repetitions: int = 3,
    detect_html_tables: bool = True
) -> DetectionResult:
    """
    Convenience function to detect repetitions in text.
    
    This function automatically selects the best detection method based on
    the content of the text.
    
    Args:
        text: The text to analyze
        min_phrase_length: Minimum length of phrase to consider
        min_repetitions: Minimum number of occurrences to flag
        detect_html_tables: Whether to use specialized HTML table detection
        
    Returns:
        DetectionResult containing all detected repetitions
    """
    if not text:
        return DetectionResult(
            has_repetition=False,
            repetitions=[],
            total_repetition_count=0,
            repetitive_content_ratio=0.0,
            original_length=0,
            detection_method="none"
        )
    
    results = []
    
    # Check for HTML content
    if detect_html_tables and ('<tr' in text.lower() or '<td' in text.lower()):
        html_detector = HTMLTableRepetitionDetector(min_repetitions)
        results.append(html_detector.detect(text))
    
    # Use consecutive detector for efficiency
    consecutive_detector = ConsecutiveRepetitionDetector(min_phrase_length, min_repetitions)
    results.append(consecutive_detector.detect(text))
    
    # Use general detector for comprehensive analysis
    general_detector = RepetitionDetector(min_phrase_length, min_repetitions)
    results.append(general_detector.detect(text))
    
    # Combine results
    all_repetitions: Dict[str, RepetitionMatch] = {}
    for result in results:
        for rep in result.repetitions:
            if rep.phrase not in all_repetitions:
                all_repetitions[rep.phrase] = rep
            elif rep.count > all_repetitions[rep.phrase].count:
                all_repetitions[rep.phrase] = rep
    
    # Filter substrings
    filtered = _filter_substrings(list(all_repetitions.values()))
    
    # Calculate combined metrics
    total_count = sum(rep.count for rep in filtered)
    repetitive_chars = sum(rep.phrase_length * (rep.count - 1) for rep in filtered)
    ratio = repetitive_chars / len(text) if text else 0.0
    
    return DetectionResult(
        has_repetition=len(filtered) > 0,
        repetitions=filtered,
        total_repetition_count=total_count,
        repetitive_content_ratio=ratio,
        original_length=len(text),
        detection_method="auto"
    )


def _filter_substrings(matches: List[RepetitionMatch]) -> List[RepetitionMatch]:
    """Remove repetitions that are substrings of larger repetitions."""
    if not matches:
        return []
    
    sorted_matches = sorted(matches, key=lambda x: x.phrase_length, reverse=True)
    
    filtered = []
    for match in sorted_matches:
        is_substring = False
        for existing in filtered:
            if match.phrase in existing.phrase:
                is_substring = True
                break
        if not is_substring:
            filtered.append(match)
    
    return filtered


def is_repetitive(
    text: str,
    threshold: float = 0.3,
    min_phrase_length: int = 20
) -> bool:
    """
    Quick check if text contains significant repetitions.
    
    Args:
        text: The text to check
        threshold: Ratio of repetitive content to flag (0.0 to 1.0)
        min_phrase_length: Minimum phrase length to consider
        
    Returns:
        True if text is considered repetitive, False otherwise
    """
    result = detect_repetition(text, min_phrase_length=min_phrase_length)
    return result.repetitive_content_ratio >= threshold


# Example usage and testing
if __name__ == "__main__":
    # Test with sample repetitive text
    sample_text = """
    <table>
    <tr><td>บริษัท ช้อยส์ จำกัด</td></tr><tr><td>ที่อยู่</td><td>18/1 หมู่ 1 ถนนศรีนครินทร์ ตำบลบางพลัด อำเภอเมือง สมุทรสาคร จ.สมุทรสาคร 74000</td></tr><tr><td>เลขที่ใบกำกับภาษี</td><td>000071131150</td></tr><tr><td>วันที่ใบกำกับภาษี</td><td>20 ธ.ค. 60</td></tr><tr><td>สถานะใบกำกับภาษี</td><td>รอตรวจสอบ</td></tr><tr><td>รหัสลูกค้า</td><td>000071131150</td></tr><tr><td>ชื่อลูกค้า</td>
    <tr><td>บริษัท ช้อยส์ จำกัด</td></tr><tr><td>ที่อยู่</td><td>18/1 หมู่ 1 ถนนศรีนครินทร์ ตำบลบางพลัด อำเภอเมือง สมุทรสาคร จ.สมุทรสาคร 74000</td></tr><tr><td>เลขที่ใบกำกับภาษี</td><td>000071131150</td></tr><tr><td>วันที่ใบกำกับภาษี</td><td>20 ธ.ค. 60</td></tr><tr><td>สถานะใบกำกับภาษี</td><td>รอตรวจสอบ</td></tr><tr><td>รหัสลูกค้า</td><td>000071131150</td></tr><tr><td>ชื่อลูกค้า</td>
    <tr><td>บริษัท ช้อยส์ จำกัด</td></tr><tr><td>ที่อยู่</td><td>18/1 หมู่ 1 ถนนศรีนครินทร์ ตำบลบางพลัด อำเภอเมือง สมุทรสาคร จ.สมุทรสาคร 74000</td></tr><tr><td>เลขที่ใบกำกับภาษี</td><td>000071131150</td></tr><tr><td>วันที่ใบกำกับภาษี</td><td>20 ธ.ค. 60</td></tr><tr><td>สถานะใบกำกับภาษี</td><td>รอตรวจสอบ</td></tr><tr><td>รหัสลูกค้า</td><td>000071131150</td></tr><tr><td>ชื่อลูกค้า</td>
    </table>
    """
    
    print("=" * 60)
    print("Repetition Detection Test")
    print("=" * 60)
    
    result = detect_repetition(sample_text)
    print(result.get_summary())
    
    print("\n" + "=" * 60)
    print("Quick Check Test")
    print("=" * 60)
    
    if is_repetitive(sample_text):
        print("⚠️  Text is flagged as REPETITIVE")
    else:
        print("✓ Text is NOT repetitive")
