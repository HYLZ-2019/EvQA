"""
Statistics analysis module for the annotation system.
Handles dashboard statistics and data analysis.
"""

import numpy as np


def get_dashboard_stats(questions_data):
    """Get comprehensive statistics for dashboard"""
    if not questions_data:
        return {"error": "No questions loaded"}
    
    # Basic statistics
    total_questions = len(questions_data)
    
    # Group by dataset
    datasets = {}
    dataset_distribution = {}
    annotator_distribution = {}
    content_type_distribution = {}
    question_type_distribution = {}
    camera_type_distribution = {}
    duration_distribution = {}
    all_durations = []
    
    for question in questions_data:
        dataset_name = question['dataset_name']
        
        # Dataset grouping
        if dataset_name not in datasets:
            datasets[dataset_name] = {
                'dataset_name': dataset_name,
                'description': question.get('description', ''),
                'annotator': question.get('annotator', ''),
                'question_count': 0
            }
        datasets[dataset_name]['question_count'] += 1
        
        # Distribution counting
        dataset_distribution[dataset_name] = dataset_distribution.get(dataset_name, 0) + 1
        
        annotator = question.get('annotator', 'Unknown')
        if annotator:
            annotator_distribution[annotator] = annotator_distribution.get(annotator, 0) + 1
        
        content_type = question.get('content_type', 'Unknown')
        if content_type:
            content_type_distribution[content_type] = content_type_distribution.get(content_type, 0) + 1
        
        question_type = question.get('question_type', 'Unknown')
        if question_type:
            question_type_distribution[question_type] = question_type_distribution.get(question_type, 0) + 1
        
        camera_type = question.get('camera_type', 'Unknown')
        if camera_type:
            camera_type_distribution[camera_type] = camera_type_distribution.get(camera_type, 0) + 1
        
        # Duration distribution (adaptive binning into 10 bins)
        duration = question.get('duration', 0)
        all_durations.append(duration)

    # Make duration distribution fast using numpy
    if all_durations:
        all_durations = np.array(all_durations)
        min_duration = np.min(all_durations)
        max_duration = np.max(all_durations)
        if min_duration == max_duration:
            duration_bins = [min_duration, max_duration + 1]
        else:
            duration_bins = np.linspace(min_duration, max_duration, 11)
        duration_distribution, _ = np.histogram(all_durations, bins=duration_bins)
        # Return ordered lists
        duration_distribution = {
            "bin_names": [f"{int(duration_bins[i])}-{int(duration_bins[i+1])}s" if i < len(duration_bins) - 1 else f"{int(duration_bins[i])}s+" for i in range(len(duration_bins))],
            "counts": duration_distribution.tolist()
        }
            
    response = {
        'total_questions': total_questions,
        'total_datasets': len(datasets),
        'datasets': list(datasets.values()),
        'dataset_distribution': dataset_distribution,
        'annotator_distribution': annotator_distribution,
        'content_type_distribution': content_type_distribution,
        'question_type_distribution': question_type_distribution,
        'camera_type_distribution': camera_type_distribution,
        'duration_distribution': duration_distribution
    }
    
    return response
