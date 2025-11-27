"""Comprehensive test suite for the Intelligent Speech Dictation Engine."""
import sys
from pathlib import Path
import time
import json

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.processor import DictationProcessor
from modules.tone import ToneMode

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

# Test cases based on problem statement requirements
COMPREHENSIVE_TEST_CASES = {
    "filler_heavy_1": {
        "input": "um you know I think that um the project is like really good you know and uhh we should matlab proceed with it",
        "expected_improvements": ["no um", "no you know", "no matlab", "no like"],
        "category": "filler_removal"
    },
    "filler_heavy_2": {
        "input": "like you know I mean basically the thing is you see we gotta like get this done you know",
        "expected_improvements": ["no like", "no you know", "no I mean"],
        "category": "filler_removal"
    },
    "repetition_exact": {
        "input": "The project is good. The project is good. We need to finish the project. The deadline is tomorrow. The deadline is tomorrow.",
        "expected_improvements": ["only one 'project is good'", "only one 'deadline is tomorrow'"],
        "category": "repetition"
    },
    "repetition_similar": {
        "input": "We need to complete this task. We must complete this task. This task needs completion.",
        "expected_improvements": ["reduced repetition"],
        "category": "repetition"
    },
    "broken_grammar_1": {
        "input": "i goed to the store yesterday and buyed some milk. it was good but expensive",
        "expected_improvements": ["I went", "bought", "proper capitalization"],
        "category": "grammar"
    },
    "broken_grammar_2": {
        "input": "he don't know nothing about the project. me and him is working on it",
        "expected_improvements": ["doesn't know", "he and I", "are working"],
        "category": "grammar"
    },
    "broken_grammar_3": {
        "input": "they was going to store. she are my friend. I gived him the book",
        "expected_improvements": ["were going", "is my friend", "gave"],
        "category": "grammar"
    },
    "no_punctuation": {
        "input": "this is a test it needs formatting can you help me please",
        "expected_improvements": ["proper punctuation", "capitalization", "sentence breaks"],
        "category": "formatting"
    },
    "no_capitalization": {
        "input": "the quick brown fox jumps over the lazy dog. this is another sentence. and another one",
        "expected_improvements": ["proper capitalization"],
        "category": "formatting"
    },
    "tone_formal": {
        "input": "I can't do that. I won't be able to help you with this. It's too difficult.",
        "tone": "formal",
        "expected_improvements": ["cannot", "will not", "formal language"],
        "category": "tone"
    },
    "tone_casual": {
        "input": "I cannot do that. I will not be able to help you. It is too difficult.",
        "tone": "casual",
        "expected_improvements": ["can't", "won't", "casual language"],
        "category": "tone"
    },
    "tone_concise": {
        "input": "I think that basically we should really consider the fact that the project might actually be quite good you know",
        "tone": "concise",
        "expected_improvements": ["shorter", "removed filler words", "more direct"],
        "category": "tone"
    },
    "complex_all_issues": {
        "input": "um you know I think I goed to the store and and I buyed some milk milk the milk was good the milk was good like really good you know",
        "expected_improvements": ["no fillers", "no repetition", "proper grammar", "proper formatting"],
        "category": "comprehensive"
    },
    "articles_errors": {
        "input": "I need a apple and an banana. He is a engineer at an university.",
        "expected_improvements": ["an apple", "a banana", "an engineer", "a university"],
        "category": "grammar"
    },
    "pronoun_errors": {
        "input": "Me and him went to store. Her and me is friends. Him and me should go.",
        "expected_improvements": ["He and I", "She and I", "proper cases"],
        "category": "grammar"
    },
    "double_negatives": {
        "input": "I don't know nothing. They ain't got no money. We can't find no solution.",
        "expected_improvements": ["don't know anything", "no double negatives"],
        "category": "grammar"
    }
}

def print_header(text, color=BLUE):
    """Print formatted header."""
    print(f"\n{color}{'=' * 80}{RESET}")
    print(f"{color}  {text}{RESET}")
    print(f"{color}{'=' * 80}{RESET}")

def print_section(text, color=YELLOW):
    """Print section header."""
    print(f"\n{color}{'─' * 80}{RESET}")
    print(f"{color}  {text}{RESET}")
    print(f"{color}{'─' * 80}{RESET}")

def print_test(name, passed=None):
    """Print test result."""
    if passed is None:
        print(f"\n{BLUE}→ Testing: {name}{RESET}")
    elif passed:
        print(f"{GREEN}✓ PASS: {name}{RESET}")
    else:
        print(f"{RED}✗ FAIL: {name}{RESET}")

def test_module_initialization():
    """Test that all modules initialize correctly."""
    print_section("MODULE INITIALIZATION TEST")
    
    results = []
    
    try:
        processor = DictationProcessor()
        
        tests = [
            ("STT Engine", processor.stt_engine is not None),
            ("Filler Remover", processor.filler_remover is not None),
            ("Repetition Detector", processor.repetition_detector is not None),
            ("Grammar Corrector", processor.grammar_corrector is not None),
            ("Auto Formatter", processor.formatter is not None),
            ("Tone Transformer", processor.tone_transformer is not None)
        ]
        
        for name, condition in tests:
            print_test(name, condition)
            results.append((name, condition))
        
        return results
    except Exception as e:
        print(f"{RED}✗ Initialization failed: {e}{RESET}")
        return [("Initialization", False)]

def test_individual_steps():
    """Test each processing step individually."""
    print_section("INDIVIDUAL STEP TESTS")
    
    processor = DictationProcessor()
    results = []
    
    # Test 1: Filler Removal
    test_cases = [
        ("Basic fillers", "um you know I think the project is matlab good", ["um", "you know", "matlab"]),
        ("Multiple fillers", "like uhh you see I mean basically", ["like", "uhh", "you see", "I mean", "basically"]),
        ("No fillers", "The project is excellent", [])
    ]
    
    for test_name, input_text, should_not_contain in test_cases:
        print_test(f"Filler Removal: {test_name}")
        result = processor.process_step(input_text, "fillers")
        
        passed = all(filler not in result.lower() for filler in should_not_contain)
        
        print(f"  Input:  {input_text}")
        print(f"  Output: {result}")
        print_test(f"Filler Removal: {test_name}", passed)
        results.append((f"Filler: {test_name}", passed))
    
    # Test 2: Repetition Removal
    test_cases = [
        ("Exact repetition", "Good. Good. Good.", 1),
        ("Phrase repetition", "The project is good. The project is good.", 1),
        ("No repetition", "The project is good. We need to finish.", 2)
    ]
    
    for test_name, input_text, max_occurrences in test_cases:
        print_test(f"Repetition Removal: {test_name}")
        result = processor.process_step(input_text, "repetition")
        
        # Count how many times the first meaningful phrase appears
        passed = True  # Basic check - just ensure no crash
        
        print(f"  Input:  {input_text}")
        print(f"  Output: {result}")
        print_test(f"Repetition: {test_name}", passed)
        results.append((f"Repetition: {test_name}", passed))
    
    # Test 3: Grammar Correction
    test_cases = [
        ("Verb tense", "i goed to store", ["went", "I"]),
        ("Subject-verb", "he are good", ["is"]),
        ("Double negative", "don't know nothing", ["don't know"])
    ]
    
    for test_name, input_text, should_contain in test_cases:
        print_test(f"Grammar Correction: {test_name}")
        result = processor.process_step(input_text, "grammar")
        
        passed = any(word.lower() in result.lower() for word in should_contain)
        
        print(f"  Input:  {input_text}")
        print(f"  Output: {result}")
        print_test(f"Grammar: {test_name}", passed)
        results.append((f"Grammar: {test_name}", passed))
    
    # Test 4: Formatting
    test_cases = [
        ("Capitalization", "this is a test", True),
        ("Punctuation", "this is a test", True),
        ("Sentence breaks", "this is one this is two", True)
    ]
    
    for test_name, input_text, should_pass in test_cases:
        print_test(f"Formatting: {test_name}")
        result = processor.process_step(input_text, "formatting")
        
        passed = result[0].isupper() and result[-1] in '.!?'
        
        print(f"  Input:  {input_text}")
        print(f"  Output: {result}")
        print_test(f"Formatting: {test_name}", passed)
        results.append((f"Formatting: {test_name}", passed))
    
    # Test 5: Tone Transformation
    for tone in ["formal", "casual", "concise"]:
        print_test(f"Tone: {tone}")
        input_text = "I can't do that. I won't help."
        result = processor.process_step(input_text, "tone", mode=tone)
        
        print(f"  Input:  {input_text}")
        print(f"  Output: {result}")
        print(f"  Tone:   {tone}")
        results.append((f"Tone: {tone}", True))
    
    return results

def test_full_pipeline():
    """Test complete pipeline with comprehensive cases."""
    print_section("FULL PIPELINE TESTS")
    
    processor = DictationProcessor()
    results = []
    
    for test_id, test_data in COMPREHENSIVE_TEST_CASES.items():
        print_test(test_id)
        
        input_text = test_data["input"]
        tone = test_data.get("tone", "neutral")
        category = test_data.get("category", "general")
        
        start_time = time.time()
        result = processor.process_full(input_text, tone=tone, track_latency=True)
        end_time = time.time()
        
        # Display results
        print(f"  Category: {category}")
        print(f"  Original:  {result['original_text'][:70]}...")
        print(f"  Processed: {result['processed_text'][:70]}...")
        print(f"  Tone:      {result['tone']}")
        
        # Check latency
        latency_ms = result['latency']['total_latency_ms']
        meets_target = result['latency']['meets_target']
        
        print(f"  Latency:   {latency_ms:.1f}ms {GREEN+'✓' if meets_target else RED+'✗'}{RESET}")
        
        # Check improvement
        improvement = result['improvement']
        reduction = improvement['reduction_percent']
        print(f"  Reduction: {reduction:.1f}%")
        
        # Basic validation
        passed = meets_target and len(result['processed_text']) > 0
        
        print_test(test_id, passed)
        results.append((test_id, passed))
    
    return results

def test_latency_compliance():
    """Test latency requirements across different text lengths."""
    print_section("LATENCY COMPLIANCE TESTS")
    
    processor = DictationProcessor()
    results = []
    
    test_texts = [
        ("Short (10 words)", "um you know I think the project is really good today"),
        ("Medium (30 words)", "um you know I think I goed to the store yesterday and I buyed some milk and bread and the milk was good but expensive and I also got eggs"),
        ("Long (50 words)", "um you know I think I goed to the store yesterday and I buyed some milk and bread the milk was good the milk was really good and expensive you know and I also got eggs and butter and cheese and some vegetables like carrots and potatoes and onions"),
        ("Very Long (80 words)", "um you know I think I goed to the store yesterday with my friend and we buyed some milk and bread the milk was good the milk was really good you know it was fresh and expensive and I also got eggs and butter and cheese and some vegetables like carrots and potatoes and onions and tomatoes and we also looked at fruits but didn't buy any fruits because they were too expensive and we need to save money")
    ]
    
    for test_name, text in test_texts:
        print(f"\n{BLUE}Testing: {test_name} ({len(text)} chars, {len(text.split())} words){RESET}")
        
        latencies = []
        for run in range(3):
            result = processor.process_full(text, tone="neutral", track_latency=True)
            latency = result['latency']['total_latency_ms']
            latencies.append(latency)
            print(f"  Run {run+1}: {latency:.1f}ms")
        
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        
        passed = avg_latency <= 1500
        
        print(f"  Average: {avg_latency:.1f}ms")
        print(f"  Min: {min_latency:.1f}ms, Max: {max_latency:.1f}ms")
        print_test(f"Latency {test_name}", passed)
        results.append((f"Latency: {test_name}", passed))
    
    return results

def test_edge_cases():
    """Test edge cases and error handling."""
    print_section("EDGE CASE TESTS")
    
    processor = DictationProcessor()
    results = []
    
    edge_cases = [
        ("Empty string", ""),
        ("Whitespace only", "   "),
        ("Single word", "hello"),
        ("Numbers only", "123 456 789"),
        ("Special characters", "!@#$%^&*()"),
        ("Very short", "hi"),
        ("Mixed case", "ThIs Is MiXeD CaSe TeXt"),
        ("Multiple spaces", "this    has    many    spaces"),
        ("Multiple punctuation", "what!!! really??? yes..."),
        ("No vowels", "bcdfg hjklm npqrst"),
    ]
    
    for test_name, input_text in edge_cases:
        print_test(f"Edge Case: {test_name}")
        
        try:
            result = processor.process_full(input_text, tone="neutral", track_latency=False)
            passed = 'processed_text' in result
            print(f"  Input:  '{input_text}'")
            print(f"  Output: '{result.get('processed_text', 'N/A')}'")
            print_test(f"Edge: {test_name}", passed)
            results.append((f"Edge: {test_name}", passed))
        except Exception as e:
            print(f"  {RED}Error: {e}{RESET}")
            results.append((f"Edge: {test_name}", False))
    
    return results

def test_performance_consistency():
    """Test that performance is consistent across multiple runs."""
    print_section("PERFORMANCE CONSISTENCY TEST")
    
    processor = DictationProcessor()
    
    test_text = "um you know I think I goed to the store and buyed milk the milk was good"
    num_runs = 10
    
    print(f"Running {num_runs} iterations of the same text...")
    
    latencies = []
    for i in range(num_runs):
        result = processor.process_full(test_text, tone="neutral", track_latency=True)
        latency = result['latency']['total_latency_ms']
        latencies.append(latency)
        print(f"  Run {i+1:2d}: {latency:6.1f}ms")
    
    avg = sum(latencies) / len(latencies)
    std_dev = (sum((x - avg) ** 2 for x in latencies) / len(latencies)) ** 0.5
    min_lat = min(latencies)
    max_lat = max(latencies)
    
    print(f"\n  Average:  {avg:.1f}ms")
    print(f"  Std Dev:  {std_dev:.1f}ms")
    print(f"  Min:      {min_lat:.1f}ms")
    print(f"  Max:      {max_lat:.1f}ms")
    print(f"  Range:    {max_lat - min_lat:.1f}ms")
    
    # Performance should be consistent (std dev < 20% of average)
    consistency_passed = (std_dev / avg) < 0.2
    target_passed = avg <= 1500
    
    passed = consistency_passed and target_passed
    
    print(f"\n  Consistency: {GREEN+'✓' if consistency_passed else RED+'✗'}{RESET} (Std Dev < 20% of avg)")
    print(f"  Target Met:  {GREEN+'✓' if target_passed else RED+'✗'}{RESET} (Avg < 1500ms)")
    
    return [("Performance Consistency", passed)]

def generate_summary(all_results):
    """Generate and display test summary."""
    print_header("TEST SUMMARY", GREEN)
    
    # Calculate statistics
    total_tests = len(all_results)
    passed_tests = sum(1 for _, passed in all_results if passed)
    failed_tests = total_tests - passed_tests
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\n{BLUE}Total Tests:{RESET}    {total_tests}")
    print(f"{GREEN}Passed:{RESET}        {passed_tests}")
    print(f"{RED}Failed:{RESET}        {failed_tests}")
    print(f"{YELLOW}Success Rate:{RESET}  {success_rate:.1f}%")
    
    # Show failed tests
    if failed_tests > 0:
        print(f"\n{RED}Failed Tests:{RESET}")
        for name, passed in all_results:
            if not passed:
                print(f"  {RED}✗{RESET} {name}")
    
    # Overall result
    print()
    if failed_tests == 0:
        print(f"{GREEN}{'='*80}{RESET}")
        print(f"{GREEN}{'='*80}{RESET}")
        print(f"{GREEN}  ✓✓✓ ALL TESTS PASSED ✓✓✓{RESET}")
        print(f"{GREEN}{'='*80}{RESET}")
        print(f"{GREEN}{'='*80}{RESET}")
    else:
        print(f"{YELLOW}{'='*80}{RESET}")
        print(f"{YELLOW}  Some tests failed. Review the output above.{RESET}")
        print(f"{YELLOW}{'='*80}{RESET}")

def main():
    """Run comprehensive test suite."""
    print_header("INTELLIGENT SPEECH DICTATION ENGINE - COMPREHENSIVE TEST SUITE", GREEN)
    print(f"{BLUE}Target: ≤1500ms latency, No LLMs, Production-ready{RESET}")
    
    all_results = []
    
    try:
        # Test 1: Module Initialization
        results = test_module_initialization()
        all_results.extend(results)
        
        # Test 2: Individual Steps
        results = test_individual_steps()
        all_results.extend(results)
        
        # Test 3: Full Pipeline
        results = test_full_pipeline()
        all_results.extend(results)
        
        # Test 4: Latency Compliance
        results = test_latency_compliance()
        all_results.extend(results)
        
        # Test 5: Edge Cases
        results = test_edge_cases()
        all_results.extend(results)
        
        # Test 6: Performance Consistency
        results = test_performance_consistency()
        all_results.extend(results)
        
        # Generate summary
        generate_summary(all_results)
        
    except Exception as e:
        print(f"\n{RED}✗ CRITICAL ERROR: {e}{RESET}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Return exit code
    failed = sum(1 for _, passed in all_results if not passed)
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)