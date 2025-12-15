"""
Data loading and management module for the annotation system.
Handles loading questions from JSON files and question data validation.
"""

import json
import os
import glob

def load_questions(question_group_name=None):
    """Load questions from all .json files in questions/ directory"""
    questions_data = []
    
    this_abs_path = os.path.abspath(__file__)
    # This is annotate_system/server_utils/data_loader.py, I want to find absolute path of annotate_system/questions
    questions_dir = os.path.join(os.path.dirname(os.path.dirname(this_abs_path)), "questions")
    if not os.path.exists(questions_dir):
        print("Error: questions/ directory not found")
        return []
    
    # Find all .json files in questions directory
    json_files = glob.glob(os.path.join(questions_dir, "*.json"))
    if not json_files:
        print("Error: No .json files found in questions/ directory")
        return []
    if question_group_name is not None:
        json_files = [f for f in json_files if question_group_name in f]
    
    print(f"Found {len(json_files)} question files: {[os.path.basename(f) for f in json_files]}")
    
    for json_file in json_files:
        try:
            questions_data.extend(load_questions_from_json_file(json_file))

        except json.JSONDecodeError as e:
            print(f"Error parsing {os.path.basename(json_file)}: {e}")
            continue
        except Exception as e:
            print(f"Error loading {os.path.basename(json_file)}: {e}")
            continue
    
    if not questions_data:
        print("Error: No valid questions loaded")
        return []
    
    print(f"Total questions loaded: {len(questions_data)}")
    return questions_data

def load_questions_from_json_file(json_file, base_directory=None):
    with open(json_file, 'r', encoding='utf-8') as f:
        question_group = json.load(f)
    
    questions_data = []
    # Validate question group format
    required_fields = ['dataset_name', 'questions']
    for field in required_fields:
        if field not in question_group:
            print(f"Warning: Missing required field '{field}' in {os.path.basename(json_file)}")
            continue
    
    dataset_name = question_group['dataset_name']
    if base_directory is None:
        base_directory = question_group['base_directory']
    
    # Process each question in the group
    for question in question_group['questions']:
        # Create a copy of the question with additional metadata
        processed_question = question.copy()
        
        # Generate unique question ID
        processed_question['unique_id'] = f"{dataset_name}_{question['question_id']}"
        
        # Convert relative file path to absolute path
        processed_question['event_path'] = os.path.join(base_directory, question['file_path'])
        
        # Add dataset metadata
        processed_question['dataset_name'] = dataset_name
        processed_question['description'] = question_group.get('description', '')
        processed_question['annotator'] = question_group.get('annotator', '')
        
        questions_data.append(processed_question)
        
    print(f"Loaded {len(question_group['questions'])} questions from {os.path.basename(json_file)}")
    return questions_data

def get_question_groups(questions_data):
    """Get organized question groups and their questions"""
    groups = {}
    
    # Organize questions by dataset
    for i, question in enumerate(questions_data):
        dataset_name = question['dataset_name']
        
        if dataset_name not in groups:
            groups[dataset_name] = {
                'name': dataset_name,
                'description': question.get('description', ''),
                'annotator': question.get('annotator', ''),
                'questions': []
            }
        
        question_info = {
            'index': i,
            'question_id': question['question_id'],
            'unique_id': question['unique_id'],
            'question': question['question'],
            'content_type': question.get('content_type', ''),
            'question_type': question.get('question_type', ''),
            'duration': question.get('duration', 0)
        }
        
        groups[dataset_name]['questions'].append(question_info)
    
    return groups


def get_overview(questions_data):
    """Get overview of all question groups"""
    if not questions_data:
        return {"error": "No questions loaded"}
    
    # Group questions by dataset
    datasets = {}
    for question in questions_data:
        dataset_name = question['dataset_name']
        if dataset_name not in datasets:
            datasets[dataset_name] = {
                'dataset_name': dataset_name,
                'description': question.get('description', ''),
                'annotator': question.get('annotator', ''),
                'question_count': 0,
                'questions': []
            }
        
        datasets[dataset_name]['question_count'] += 1
        datasets[dataset_name]['questions'].append({
            'unique_id': question['unique_id'],
            'question_id': question['question_id'],
            'question': question['question'],
            'content_type': question.get('content_type', ''),
            'question_type': question.get('question_type', ''),
            'duration': question.get('duration', 0)
        })
    
    overview = {
        'total_datasets': len(datasets),
        'total_questions': len(questions_data),
        'datasets': list(datasets.values())
    }
    
    return overview
