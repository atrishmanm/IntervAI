"""
data_pipeline/build_concept_graph.py
=====================================
Builds a CS concept graph with prerequisite relationships,
common mistakes, and follow-up topics.

Output: data/processed/concept_graph.json
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"


CONCEPT_GRAPH = {
    "arrays": {
        "display_name": "Arrays",
        "prerequisites": [],
        "concepts": [
            "contiguous_memory",
            "index_access",
            "fixed_size",
            "O(1)_random_access",
        ],
        "complexity": {"access": "O(1)", "search": "O(n)", "insert": "O(n)"},
        "common_mistakes": [
            "off_by_one_errors",
            "accessing_out_of_bounds",
            "assuming_dynamic_size",
        ],
        "related": ["linked_lists", "hash_tables"],
        "followups": [
            "What is the difference between an array and a linked list?",
            "When would you use an array over a linked list?",
            "How does a dynamic array (like Python's list) work?",
        ],
        "difficulty": "easy",
        "subtopics": ["1d_arrays", "2d_arrays", "dynamic_arrays", "sparse_arrays"],
    },
    "linked_lists": {
        "display_name": "Linked Lists",
        "prerequisites": ["arrays"],
        "concepts": [
            "nodes",
            "pointers",
            "singly_linked",
            "doubly_linked",
            "O(1)_insertion_deletion",
        ],
        "complexity": {"access": "O(n)", "search": "O(n)", "insert": "O(1)"},
        "common_mistakes": [
            "losing_head_reference",
            "not_handling_null",
            "infinite_loop_in_traversal",
        ],
        "related": ["arrays", "stacks", "queues"],
        "followups": [
            "How do you detect a cycle in a linked list?",
            "How do you reverse a linked list?",
            "What is the difference between singly and doubly linked lists?",
        ],
        "difficulty": "easy",
        "subtopics": ["singly", "doubly", "circular", "skip_list"],
    },
    "stacks": {
        "display_name": "Stacks",
        "prerequisites": ["arrays", "linked_lists"],
        "concepts": ["LIFO", "push", "pop", "peek", "stack_overflow"],
        "complexity": {"push": "O(1)", "pop": "O(1)", "peek": "O(1)"},
        "common_mistakes": [
            "using_for_FIFO_tasks",
            "not_checking_empty_stack",
            "stack_overflow_from_recursion",
        ],
        "related": ["queues", "recursion", "expression_parsing"],
        "followups": [
            "How would you implement a stack using arrays vs linked lists?",
            "What real-world problems use stacks?",
            "How do you evaluate a postfix expression using a stack?",
        ],
        "difficulty": "easy",
        "subtopics": ["array_impl", "linked_list_impl", "applications"],
    },
    "queues": {
        "display_name": "Queues",
        "prerequisites": ["arrays", "linked_lists"],
        "concepts": ["FIFO", "enqueue", "dequeue", "front", "rear"],
        "complexity": {"enqueue": "O(1)", "dequeue": "O(1)", "front": "O(1)"},
        "common_mistakes": [
            "confusing_with_stack",
            "not_circular_queue_implementation",
            "priority_queue_confusion",
        ],
        "related": ["stacks", "graph_traversal", "scheduling"],
        "followups": [
            "What is a circular queue and why is it useful?",
            "What is a priority queue and when would you use one?",
            "How does BFS use a queue?",
        ],
        "difficulty": "easy",
        "subtopics": ["simple_queue", "circular_queue", "deque", "priority_queue"],
    },
    "hash_tables": {
        "display_name": "Hash Tables",
        "prerequisites": ["arrays"],
        "concepts": [
            "hash_function",
            "buckets",
            "collision_handling",
            "chaining",
            "open_addressing",
            "load_factor",
        ],
        "complexity": {
            "average_lookup": "O(1)",
            "worst_lookup": "O(n)",
            "insert": "O(1)",
        },
        "common_mistakes": [
            "ignoring_collisions",
            "using_bad_hash_function",
            "not_resizing",
            "confusing_with_O(1)_guarantee",
        ],
        "related": ["arrays", "trees", "sets"],
        "followups": [
            "What happens when two keys hash to the same index?",
            "What is the difference between chaining and open addressing?",
            "Why is hash table lookup O(1) on average but O(n) worst case?",
        ],
        "difficulty": "medium",
        "subtopics": ["hash_functions", "collision_resolution", "load_factor", "rehashing"],
    },
    "binary_search": {
        "display_name": "Binary Search",
        "prerequisites": ["arrays", "time_complexity"],
        "concepts": [
            "sorted_array",
            "divide_and_conquer",
            "midpoint",
            "search_space_reduction",
            "O(log_n)",
        ],
        "complexity": {"time": "O(log n)", "space": "O(1)"},
        "common_mistakes": [
            "unsorted_input",
            "off_by_one",
            "infinite_loop",
            "wrong_midpoint_calculation",
        ],
        "related": ["linear_search", "interpolation_search", "two_pointers"],
        "followups": [
            "Why must the array be sorted?",
            "How do you handle duplicates?",
            "What about rotated sorted arrays?",
            "How would you implement it recursively?",
        ],
        "difficulty": "medium",
        "subtopics": ["iterative", "recursive", "variants", "lower_upper_bound"],
    },
    "sorting_basics": {
        "display_name": "Sorting Algorithms",
        "prerequisites": ["recursion", "time_complexity"],
        "concepts": [
            "comparison_sort",
            "divide_and_conquer",
            "in_place",
            "stable_sort",
            "merge_sort",
            "quick_sort",
            "heap_sort",
        ],
        "complexity": {
            "bubble": "O(n^2)",
            "merge": "O(n log n)",
            "quick_avg": "O(n log n)",
            "quick_worst": "O(n^2)",
        },
        "common_mistakes": [
            "confusing_stability",
            "not_understanding_quick_sort_worst_case",
            "choosing_wrong_algorithm_for_constraints",
        ],
        "related": ["binary_search", "heap_sort", "counting_sort", "radix_sort"],
        "followups": [
            "When would you use merge sort over quick sort?",
            "What makes a sorting algorithm stable?",
            "How does quicksort achieve O(n log n) on average?",
        ],
        "difficulty": "medium",
        "subtopics": ["bubble", "selection", "insertion", "merge", "quick", "heap"],
    },
    "trees": {
        "display_name": "Trees",
        "prerequisites": ["recursion", "linked_lists"],
        "concepts": [
            "nodes",
            "root",
            "children",
            "leaf",
            "height",
            "depth",
            "binary_tree",
            "BST",
        ],
        "complexity": {
            "search_bst_avg": "O(log n)",
            "search_bst_worst": "O(n)",
            "traversal": "O(n)",
        },
        "common_mistakes": [
            "confusing_height_and_depth",
            "not_understanding_BST_property",
            "off_by_one_in_height_calculation",
        ],
        "related": ["binary_trees", "heaps", "tries", "graphs"],
        "followups": [
            "What is the difference between a tree and a graph?",
            "What are the different tree traversal orders?",
            "When does a BST become degenerate?",
        ],
        "difficulty": "medium",
        "subtopics": ["binary_tree", "BST", "balanced_trees", "traversals"],
    },
    "binary_trees": {
        "display_name": "Binary Trees",
        "prerequisites": ["trees", "recursion"],
        "concepts": [
            "left_child",
            "right_child",
            "complete_tree",
            "perfect_tree",
            "balanced_tree",
            "height",
        ],
        "complexity": {"traversal": "O(n)", "height": "O(n) worst, O(log n) balanced"},
        "common_mistakes": [
            "not_base_case_in_recursion",
            "confusing_complete_and_full",
            "not_handling_single_node",
        ],
        "related": ["trees", "heaps", "BST"],
        "followups": [
            "How do you check if a binary tree is balanced?",
            "What is the difference between a complete and perfect binary tree?",
            "How do you find the lowest common ancestor?",
        ],
        "difficulty": "medium",
        "subtopics": ["complete", "perfect", "balanced", "degenerate"],
    },
    "heaps": {
        "display_name": "Heaps",
        "prerequisites": ["binary_trees", "arrays"],
        "concepts": [
            "max_heap",
            "min_heap",
            "heap_property",
            "complete_binary_tree",
            "heapify",
        ],
        "complexity": {"insert": "O(log n)", "extract": "O(log n)", "peek": "O(1)"},
        "common_mistakes": [
            "confusing_with_BST",
            "not_0_indexed_correctly",
            "forgetting_to_heapify_after_modification",
        ],
        "related": ["priority_queue", "heap_sort", "graphs"],
        "followups": [
            "How do you build a heap from an unsorted array?",
            "How is a heap different from a BST?",
            "How does heap sort work?",
        ],
        "difficulty": "medium",
        "subtopics": ["max_heap", "min_heap", "heapify", "heap_sort"],
    },
    "graphs": {
        "display_name": "Graphs",
        "prerequisites": ["trees", "queues", "stacks"],
        "concepts": [
            "vertices",
            "edges",
            "directed",
            "undirected",
            "weighted",
            "adjacency_matrix",
            "adjacency_list",
        ],
        "complexity": {
            "adj_matrix_access": "O(1)",
            "adj_list_traversal": "O(V+E)",
        },
        "common_mistakes": [
            "confusing_matrix_and_list_representations",
            "not_handling_disconnected_components",
            "infinite_loop_in_cyclic_graphs",
        ],
        "related": ["graph_traversal", "trees", "dynamic_programming"],
        "followups": [
            "When would you use an adjacency matrix vs adjacency list?",
            "How do you detect a cycle in a graph?",
            "What is the difference between BFS and DFS?",
        ],
        "difficulty": "hard",
        "subtopics": ["representation", "directed", "weighted", "special_graphs"],
    },
    "graph_traversal": {
        "display_name": "Graph Traversal",
        "prerequisites": ["graphs", "queues", "stacks"],
        "concepts": [
            "BFS",
            "DFS",
            "queue_for_BFS",
            "stack_recursion_for_DFS",
            "shortest_path_unweighted",
        ],
        "complexity": {"BFS": "O(V+E)", "DFS": "O(V+E)"},
        "common_mistakes": [
            "confusing_BFS_and_DFS",
            "not_marking_visited",
            "using_DFS_for_shortest_path",
        ],
        "related": ["graphs", "trees", "dynamic_programming"],
        "followups": [
            "When would you use BFS over DFS?",
            "How do you find the shortest path in an unweighted graph?",
            "How do you detect a cycle using DFS?",
        ],
        "difficulty": "hard",
        "subtopics": ["BFS", "DFS", "topological_sort", "connected_components"],
    },
    "dynamic_programming": {
        "display_name": "Dynamic Programming",
        "prerequisites": ["recursion", "arrays", "time_complexity"],
        "concepts": [
            "overlapping_subproblems",
            "optimal_substructure",
            "memoization",
            "tabulation",
            "state_transition",
        ],
        "complexity": {"varies": "depends on problem and state space"},
        "common_mistakes": [
            "not_identifying_overlapping_subproblems",
            "wrong_state_transition",
            "not_base_case",
            "choosing_wrong_data_structure_for_tabulation",
        ],
        "related": ["recursion", "greedy_algorithms", "graphs"],
        "followups": [
            "How do you identify if a problem can be solved with DP?",
            "What is the difference between top-down and bottom-up DP?",
            "Can you solve this problem with both memoization and tabulation?",
        ],
        "difficulty": "hard",
        "subtopics": ["1d_dp", "2d_dp", "knapsack", "lcs", "lis", "matrix_chain"],
    },
    "greedy_algorithms": {
        "display_name": "Greedy Algorithms",
        "prerequisites": ["sorting_basics", "time_complexity"],
        "concepts": [
            "greedy_choice_property",
            "optimal_substructure",
            "local_optimum",
            "global_optimum",
        ],
        "complexity": {"varies": "depends on sorting and selection"},
        "common_mistakes": [
            "assuming_greedy_always_works",
            "not_proving_greedy_choice",
            "confusing_with_DP",
        ],
        "related": ["dynamic_programming", "sorting_basics", "graphs"],
        "followups": [
            "When does a greedy algorithm fail?",
            "How do you prove a greedy algorithm is correct?",
            "What is the difference between greedy and DP?",
        ],
        "difficulty": "hard",
        "subtopics": ["activity_selection", "huffman", "dijkstra", "kruskal"],
    },
    "recursion": {
        "display_name": "Recursion",
        "prerequisites": [],
        "concepts": [
            "base_case",
            "recursive_case",
            "call_stack",
            "tail_recursion",
            "stack_overflow",
        ],
        "complexity": {"depends": "on number of recursive calls and work per call"},
        "common_mistakes": [
            "missing_base_case",
            "infinite_recursion",
            "excessive_stack_depth",
            "not_optimizing_redundant_calls",
        ],
        "related": ["dynamic_programming", "trees", "backtracking"],
        "followups": [
            "How do you convert recursion to iteration?",
            "What is tail recursion and why does it matter?",
            "When should you use memoization with recursion?",
        ],
        "difficulty": "easy",
        "subtopics": ["linear_recursion", "tree_recursion", "tail_recursion", "mutual_recursion"],
    },
    "time_complexity": {
        "display_name": "Time Complexity",
        "prerequisites": [],
        "concepts": [
            "Big_O",
            "Big_Theta",
            "Big_Omega",
            "constant",
            "logarithmic",
            "linear",
            "quadratic",
            "exponential",
        ],
        "complexity": {},
        "common_mistakes": [
            "confusing_best_and_worst_case",
            "not_accounting_for_all_operations",
            "confusing_O(n)_with_O(2n)",
        ],
        "related": ["space_complexity", "sorting_basics", "searching"],
        "followups": [
            "What is the difference between O(n) and O(n log n)?",
            "How do you analyze the time complexity of nested loops?",
            "What is amortized analysis?",
        ],
        "difficulty": "easy",
        "subtopics": ["big_o", "amortized", "best_worst_average", "space_complexity"],
    },
    "space_complexity": {
        "display_name": "Space Complexity",
        "prerequisites": ["time_complexity"],
        "concepts": [
            "auxiliary_space",
            "input_space",
            "stack_space",
            "in_place",
            "O(1)_space",
        ],
        "complexity": {},
        "common_mistakes": [
            "forgetting_recursion_stack_space",
            "confusing_input_and_auxiliary_space",
            "not_accounting_for_data_structure_overhead",
        ],
        "related": ["time_complexity", "in_place_algorithms"],
        "followups": [
            "What is the space complexity of merge sort?",
            "How do you reduce space complexity of an algorithm?",
            "What does 'in-place' mean?",
        ],
        "difficulty": "easy",
        "subtopics": ["auxiliary", "recursive_stack", "data_structures"],
    },
    "two_pointers": {
        "display_name": "Two Pointers",
        "prerequisites": ["arrays", "time_complexity"],
        "concepts": [
            "opposite_ends",
            "same_direction",
            "fast_slow_pointers",
            "pair_sum",
            "palindrome_check",
        ],
        "complexity": {"time": "O(n)", "space": "O(1)"},
        "common_mistakes": [
            "not_terminating_correctly",
            "moving_both_pointers_wrong",
            "not_sorted_input_when_required",
        ],
        "related": ["sliding_window", "binary_search"],
        "followups": [
            "When do you use two pointers moving in opposite directions?",
            "How do you detect a cycle with fast and slow pointers?",
            "What problems can be solved with two pointers?",
        ],
        "difficulty": "medium",
        "subtopics": ["opposite_ends", "same_direction", "fast_slow"],
    },
    "sliding_window": {
        "display_name": "Sliding Window",
        "prerequisites": ["arrays", "hash_tables"],
        "concepts": [
            "fixed_window",
            "variable_window",
            "window_expansion",
            "window_shrinkage",
            "running_sum",
        ],
        "complexity": {"time": "O(n)", "space": "O(1) to O(k)"},
        "common_mistakes": [
            "not_shrinking_window_correctly",
            "off_by_one_in_window_size",
            "not_handling_empty_window",
        ],
        "related": ["two_pointers", "hash_tables"],
        "followups": [
            "How do you find the maximum sum subarray of size k?",
            "When do you use a fixed vs variable window?",
            "How do you find the longest substring without repeating characters?",
        ],
        "difficulty": "medium",
        "subtopics": ["fixed_size", "variable_size", "concatenation"],
    },
    "tries": {
        "display_name": "Tries",
        "prerequisites": ["trees", "strings"],
        "concepts": [
            "prefix_tree",
            "character_nodes",
            "end_of_word",
            "autocomplete",
            "prefix_search",
        ],
        "complexity": {"search": "O(m)", "insert": "O(m)", "note": "m = key length"},
        "common_mistakes": [
            "not_marking_end_of_word",
            "memory_inefficient_implementation",
            "not_handling_deletion",
        ],
        "related": ["hash_tables", "trees"],
        "followups": [
            "How does a trie differ from a hash table for string lookup?",
            "How do you implement autocomplete using a trie?",
            "What is a compressed trie (radix tree)?",
        ],
        "difficulty": "hard",
        "subtopics": ["basic", "compressed", "suffix_trie", "applications"],
    },
}


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "concept_graph.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(CONCEPT_GRAPH, f, indent=2, ensure_ascii=False)

    print(f"Concept graph saved to {out_path}")
    print(f"  Total concepts: {len(CONCEPT_GRAPH)}")

    # Print summary
    for name, data in CONCEPT_GRAPH.items():
        prereqs = ", ".join(data["prerequisites"]) if data["prerequisites"] else "none"
        print(f"  {data['display_name']:25s} prereqs: {prereqs}")


if __name__ == "__main__":
    main()
