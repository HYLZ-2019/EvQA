"""
Video caching module for the annotation system.
Handles caching of generated videos to improve performance.
"""

import os
import shutil


def get_cache_path(dataset_name, question_id, video_type):
    """
    Get the cache path for a specific video.
    
    Args:
        dataset_name (str): Name of the dataset (e.g., "PEDRo", "eventvot")
        question_id (str): Question ID (e.g., "rec18_00000_00019")
        video_type (str): Type of video ("vis" or "e2vid")
    
    Returns:
        str: Full path to the cached video file
    """
    # Create cache directory structure: annotate_system/video_cache/{video_type}/{dataset_name}/{question_id}.mp4
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_dir = os.path.join(script_dir, "video_cache", video_type, dataset_name)
    
    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)
    
    # Return full path to cached video
    return os.path.join(cache_dir, f"{question_id}.mp4")


def is_video_cached(dataset_name, question_id, video_type):
    """
    Check if a video is already cached.
    
    Args:
        dataset_name (str): Name of the dataset
        question_id (str): Question ID
        video_type (str): Type of video ("vis" or "e2vid")
    
    Returns:
        bool: True if video is cached and valid, False otherwise
    """
    cache_path = get_cache_path(dataset_name, question_id, video_type)
    
    # Check if file exists and has content
    if os.path.exists(cache_path):
        file_size = os.path.getsize(cache_path)
        if file_size > 0:
            print(f"Found cached video: {cache_path}, size: {file_size} bytes")
            return True
        else:
            print(f"Cached video file is empty, removing: {cache_path}")
            try:
                os.unlink(cache_path)
            except:
                pass
    
    return False


def get_cached_video_path(dataset_name, question_id, video_type):
    """
    Get the path to a cached video if it exists.
    
    Args:
        dataset_name (str): Name of the dataset
        question_id (str): Question ID
        video_type (str): Type of video ("vis" or "e2vid")
    
    Returns:
        str: Path to cached video if it exists and is valid, None otherwise
    """
    if is_video_cached(dataset_name, question_id, video_type):
        return get_cache_path(dataset_name, question_id, video_type)
    return None


def clear_cache(dataset_name=None, video_type=None):
    """
    Clear video cache.
    
    Args:
        dataset_name (str, optional): Clear cache for specific dataset only
        video_type (str, optional): Clear cache for specific video type only
    """
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_root = os.path.join(script_dir, "video_cache")
    
    if not os.path.exists(cache_root):
        print("No cache directory found")
        return
    
    if video_type and dataset_name:
        # Clear specific dataset and video type
        cache_path = os.path.join(cache_root, video_type, dataset_name)
        if os.path.exists(cache_path):
            shutil.rmtree(cache_path)
            print(f"Cleared cache for {video_type}/{dataset_name}")
    elif video_type:
        # Clear specific video type
        cache_path = os.path.join(cache_root, video_type)
        if os.path.exists(cache_path):
            shutil.rmtree(cache_path)
            print(f"Cleared cache for {video_type}")
    elif dataset_name:
        # Clear specific dataset (all video types)
        for vtype in ["vis", "e2vid"]:
            cache_path = os.path.join(cache_root, vtype, dataset_name)
            if os.path.exists(cache_path):
                shutil.rmtree(cache_path)
                print(f"Cleared cache for {vtype}/{dataset_name}")
    else:
        # Clear all cache
        shutil.rmtree(cache_root)
        print("Cleared all video cache")


def get_cache_stats():
    """
    Get statistics about the video cache.
    
    Returns:
        dict: Cache statistics including file counts and sizes
    """
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_root = os.path.join(script_dir, "video_cache")
    
    stats = {
        "total_files": 0,
        "total_size_bytes": 0,
        "by_type": {}
    }
    
    if not os.path.exists(cache_root):
        return stats
    
    for video_type in ["vis", "e2vid"]:
        type_dir = os.path.join(cache_root, video_type)
        if not os.path.exists(type_dir):
            continue
            
        type_stats = {
            "files": 0,
            "size_bytes": 0,
            "datasets": {}
        }
        
        for dataset_name in os.listdir(type_dir):
            dataset_dir = os.path.join(type_dir, dataset_name)
            if not os.path.isdir(dataset_dir):
                continue
                
            dataset_stats = {
                "files": 0,
                "size_bytes": 0
            }
            
            for filename in os.listdir(dataset_dir):
                file_path = os.path.join(dataset_dir, filename)
                if os.path.isfile(file_path) and filename.endswith('.mp4'):
                    file_size = os.path.getsize(file_path)
                    dataset_stats["files"] += 1
                    dataset_stats["size_bytes"] += file_size
                    type_stats["files"] += 1
                    type_stats["size_bytes"] += file_size
                    stats["total_files"] += 1
                    stats["total_size_bytes"] += file_size
            
            if dataset_stats["files"] > 0:
                type_stats["datasets"][dataset_name] = dataset_stats
        
        if type_stats["files"] > 0:
            stats["by_type"][video_type] = type_stats
    
    return stats
