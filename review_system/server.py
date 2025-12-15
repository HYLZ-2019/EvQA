import os
import sys
import tempfile
import glob
from flask import Flask, render_template, request, jsonify, send_file
import argparse

# Add the parent directory to sys.path to enable imports from sibling directories
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Import utilities from server_utils
from server_utils.data_loader import load_questions, get_question_groups, get_overview, load_questions_from_json_file
from server_utils.video_processor import h5_to_videos
from server_utils.user_manager import (
    get_answered_question_ids, get_user_wrong_answers, get_next_unanswered_question_index,
    save_log, check_all_completed, get_user_stats, get_user_progress
)
from server_utils.stats_analyzer import get_dashboard_stats
from server_utils.video_cache import (
    get_cached_video_path, get_cache_path, is_video_cached, clear_cache, get_cache_stats
)

# Import the VideoReconstructor
from adaptive_e2vid import VideoReconstructor

'''
The server is used for remote data annotation.
It will read from all .json files in the questions/ directory. Each file contains a question group with multiple choice questions. 
The server will visualize the event data into videos and serve them through a web interface. 
The user can enter/modify their username on any page (automatically saved in browser). 
The user can choose the correct answer for each question. 
A log file will be saved in logs/{date}_{username}.json, saving the username, question, and answer.
'''

app = Flask(__name__)

# Global variables
questions_data = []
# Removed global current_question_index, username, manual_navigation to avoid multi-user conflicts

# Initialize the reconstructor globally for reuse
#reconstructor = VideoReconstructor()

@app.route('/')
def index():
    """Main page - username entry and question display"""
    return render_template('index.html')

@app.route('/set_username', methods=['POST'])
def set_username():
	"""Set the username for the session and save to local storage"""
	new_username = request.json.get('username')
	if new_username:
		username = new_username.strip()
		return jsonify({"status": "success", "username": username})
	else:
		return jsonify({"error": "Username cannot be empty"}), 400

@app.route('/set_username_and_get_question', methods=['POST'])
def set_username_and_get_question():
    """Set username and automatically navigate to next unanswered question"""
    new_username = request.json.get('username')
    if not new_username:
        return jsonify({"error": "Username cannot be empty"}), 400
    
    username = new_username.strip()
    
    # Find the next unanswered question for this user
    next_index = get_next_unanswered_question_index(username, questions_data)
    
    # Get user progress
    answered_ids = get_answered_question_ids(username, questions_data)
    all_completed = check_all_completed(username, questions_data)
    
    print(f"User {username} set, directed to question index {next_index}")
    
    return jsonify({
        "status": "success", 
        "username": username,
        "next_question_index": next_index,
        "next_question_id": questions_data[next_index]['question_id'] if next_index < len(questions_data) else None,
        "completed_count": len(answered_ids),
        "total_questions": len(questions_data),
        "all_completed": all_completed
    })

@app.route('/get_username')
def get_username():
	"""Get current username - Note: this only returns empty since we removed global username"""
	return jsonify({"username": ""})

@app.route('/get_question')
def get_question():
    """Get current question data, automatically advancing to next unanswered question"""
    # Get parameters from request
    username = request.args.get('username', '').strip()
    target_question_id = request.args.get('target')  # Specific question to jump to
    question_index = request.args.get('index', type=int)  # Specific index to jump to
    
    # Determine which question to show
    if target_question_id:
        # Jump to specific question by ID
        current_question_index = None
        for i, question in enumerate(questions_data):
            if question['question_id'] == target_question_id:
                current_question_index = i
                print(f"Jumping to question ID {target_question_id} at index {i}")
                break
        
        if current_question_index is None:
            return jsonify({"error": f"Question with ID '{target_question_id}' not found"}), 404
            
    elif question_index is not None:
        # Jump to specific question by index
        if 0 <= question_index < len(questions_data):
            current_question_index = question_index
            print(f"Jumping to question index {question_index}")
        else:
            return jsonify({"error": "Invalid question index"}), 400
            
    elif username:
        # Auto-navigate to next unanswered question for user
        current_question_index = get_next_unanswered_question_index(username, questions_data)
        print(f"User {username} directed to question index {current_question_index}")
    else:
        # No username or target specified, default to first question
        current_question_index = 0
        print("No username specified, showing first question")
    
    if current_question_index < len(questions_data):
        question = questions_data[current_question_index]
        
        # Check if this question has already been answered by the user
        is_answered = False
        if username:
            answered_ids = get_answered_question_ids(username, questions_data)
            is_answered = question['question_id'] in answered_ids
        
        # Create response with additional metadata for display
        response = {
            'unique_id': question['unique_id'],
            'question_id': question['question_id'],
            'dataset_name': question['dataset_name'],
            'description': question.get('description', ''),
            'annotator': question.get('annotator', ''),
            'question': question['question'],
            'answer': question['answer'],
            'choices': question.get('choices', []),
            'choices_cn': question.get('choices_cn', []),
            'wrong_answers': question.get('wrong_answers', []),
            'question_cn': question.get('question_cn', ''),
            'answer_cn': question.get('answer_cn', ''),
            'wrong_answers_cn': question.get('wrong_answers_cn', []),
            'has_images': question.get('has_images', False),
            'camera_type': question.get('camera_type', ''),
            'resolution': question.get('resolution', ''),
            'content_type': question.get('content_type', ''),
            'question_type': question.get('question_type', ''),
            'duration': question.get('duration', 0),
            'current_index': current_question_index + 1,
            'total_questions': len(questions_data),
            'is_answered': is_answered,
            'is_release': question.get('is_release', False) or (len(question.get('choices', [])) > 0)
        }
        return jsonify(response)
    else:
        return jsonify({"message": "All questions completed"})

@app.route('/get_video')
def get_video():
    """Generate and serve video for specified question"""
    # Get parameters from request
    username = request.args.get('username', '').strip()
    target_question_id = request.args.get('target')  # Specific question to get video for
    question_index = request.args.get('index', type=int)  # Specific index to get video for
    video_type = request.args.get('type', 'vis')  # Type of video: 'vis' or 'e2vid'
    
    # Determine which question's video to serve
    if target_question_id:
        # Find question by ID
        current_question_index = None
        for i, question in enumerate(questions_data):
            if question['question_id'] == target_question_id:
                current_question_index = i
                break
        if current_question_index is None:
            return "Question not found", 404
            
    elif question_index is not None:
        # Use specific index
        if 0 <= question_index < len(questions_data):
            current_question_index = question_index
        else:
            return "Invalid question index", 400
            
    elif username:
        # Auto-navigate to next unanswered question for user
        current_question_index = get_next_unanswered_question_index(username, questions_data)
    else:
        # Default to first question
        current_question_index = 0
    
    if current_question_index < len(questions_data):
        question = questions_data[current_question_index]
        h5_path = question['event_path']
        dataset_name = question['dataset_name']
        question_id = question['question_id']
        
        print(f"Generating {video_type} video for question {current_question_index}, dataset: {dataset_name}, question_id: {question_id}")
        
        # Check if video is already cached
        cached_video_path = get_cached_video_path(dataset_name, question_id, video_type)
        if cached_video_path:
            print(f"Using cached video: {cached_video_path}")
            try:
                # Verify cached file exists and has content before serving
                if not os.path.exists(cached_video_path):
                    print("Error: Cached video file does not exist")
                    return "Cached video file not found", 500
                
                file_size = os.path.getsize(cached_video_path)
                print(f"Serving cached video file: {cached_video_path}, size: {file_size} bytes")
                
                if file_size == 0:
                    print("Error: Cached video file is empty")
                    return "Cached video file is empty", 500
                
                # Add proper headers for video streaming
                response = send_file(
                    cached_video_path, 
                    as_attachment=False, 
                    mimetype='video/mp4',
                    download_name=f'{video_type}_video.mp4'
                )
                response.headers['Accept-Ranges'] = 'bytes'
                response.headers['Content-Type'] = 'video/mp4'
                response.headers['Cache-Control'] = 'no-cache'
                
                return response
            except Exception as e:
                print(f"Error serving cached video file: {e}")
                return "Error serving cached video", 500
        
        # Video not cached, generate new one directly to cache
        print(f"Generating new {video_type} video...")
        cache_video_path = get_cache_path(dataset_name, question_id, video_type)
        
        # Generate the requested video type directly to cache
        success = False
        if video_type == 'e2vid':
            # Generate e2vid video (requires reconstructor)
            reconstructor = VideoReconstructor() # 照理来说应该全局共用一个的，但是那样总是爆显存，不知道为什么；放到这里再初始化就好了
            success = h5_to_videos(h5_path, e2vid_output_path=cache_video_path, reconstructor=reconstructor)
        else:
            # Generate vis video by default
            success = h5_to_videos(h5_path, vis_output_path=cache_video_path)
        
        if success:
            try:
                # Verify file exists and has content before serving
                if not os.path.exists(cache_video_path):
                    print("Error: Video file does not exist")
                    return "Video file not found", 500
                
                file_size = os.path.getsize(cache_video_path)
                print(f"Generated and serving video file: {cache_video_path}, size: {file_size} bytes")
                
                if file_size == 0:
                    print("Error: Video file is empty")
                    return "Video file is empty", 500
                
                # Add proper headers for video streaming
                response = send_file(
                    cache_video_path, 
                    as_attachment=False, 
                    mimetype='video/mp4',
                    download_name=f'{video_type}_video.mp4'
                )
                response.headers['Accept-Ranges'] = 'bytes'
                response.headers['Content-Type'] = 'video/mp4'
                response.headers['Cache-Control'] = 'no-cache'
                
                # No need to clean up cache file after response
                return response
            except Exception as e:
                print(f"Error serving video file: {e}")
                # Clean up on error
                try:
                    if os.path.exists(cache_video_path):
                        os.unlink(cache_video_path)
                except:
                    pass
                return "Error serving video", 500
        else:
            # Clean up on error
            try:
                if os.path.exists(cache_video_path):
                    os.unlink(cache_video_path)
            except:
                pass
            return f"Error generating {video_type} video", 500
    return "No question available", 404

@app.route('/submit_answer', methods=['POST'])
def submit_answer():
    """Submit answer for specified question"""
    # Get data from request
    data = request.json
    username = data.get('username', '').strip()
    answer = data.get('answer')
    question_id = data.get('question_id')  # The specific question being answered
    
    if not username:
        username = "anonymous"
    
    if not answer:
        return jsonify({"error": "Answer is required"}), 400
    
    if not question_id:
        return jsonify({"error": "Question ID is required"}), 400
    
    # Find the question data
    question_data = None
    for q in questions_data:
        if q['question_id'] == question_id:
            question_data = q
            break
    
    if not question_data:
        return jsonify({"error": "Question not found"}), 404
    
    # Save to log
    save_log(username, question_id, answer)
    print(f"User {username} answered question {question_id}: {answer}")
    
    # Check if user has completed all questions after this submission
    all_completed = check_all_completed(username, questions_data)
    
    # Find next unanswered question
    next_index = get_next_unanswered_question_index(username, questions_data)
    next_question_id = questions_data[next_index]['question_id'] if next_index < len(questions_data) else None
    
    # Determine if there are more questions to answer
    next_available = not all_completed and next_index < len(questions_data)
    
    return jsonify({
        "status": "success", 
        "next_available": next_available,
        "username": username,
        "all_completed": all_completed,
        "next_question_index": next_index,
        "next_question_id": next_question_id
    })

@app.route('/reset')
def reset():
    """Reset to first unanswered question for the current user"""
    global current_question_index, username
    
    # Get username from request args if needed
    request_username = request.args.get('username')
    if request_username:
        username = request_username.strip()
    
    if username:
        # Reset to first unanswered question
        current_question_index = get_next_unanswered_question_index(username, questions_data)
        print(f"Reset: User {username} directed to question index {current_question_index}")
    else:
        # No username, reset to first question
        current_question_index = 0
        print("Reset: No username, reset to first question")
    
    return jsonify({
        "status": "reset", 
        "current_index": current_question_index,
        "username": username or ""
    })


@app.route('/get_overview')
def get_overview_route():
    """Get overview of all question groups"""
    return jsonify(get_overview(questions_data))


@app.route('/get_question_groups')
def get_question_groups_route():
    """Get organized question groups and their questions"""
    return jsonify(get_question_groups(questions_data))

@app.route('/set_question', methods=['POST'])
def set_question():
	"""Set current question by index"""
	global current_question_index
	question_index = request.json.get('index')
	
	if question_index is not None and 0 <= question_index < len(questions_data):
		current_question_index = question_index
		return jsonify({"status": "success", "current_index": current_question_index})
	else:
		return jsonify({"error": "Invalid question index"}), 400

@app.route('/get_user_progress')
def get_user_progress_route():
    """Get user's progress - which questions they have answered"""
    # Get username from request args (required)
    username = request.args.get('username', '').strip()
    
    if not username:
        return jsonify({
            "error": "Username is required",
            "answered_questions": [], 
            "total_questions": len(questions_data), 
            "completed_count": 0, 
            "all_completed": False,
            "wrong_answers": [],
            "total_wrong": 0
        })
    
    return jsonify(get_user_progress(username, questions_data))


@app.route('/get_user_wrong_answers')
def get_user_wrong_answers_api():
    """Get user's wrong answers - questions they answered incorrectly"""
    # Get username from request args (required)
    username = request.args.get('username', '').strip()
    
    if not username:
        return jsonify({"wrong_answers": [], "total_wrong": 0, "error": "Username is required"})
    
    wrong_answers = get_user_wrong_answers(username, questions_data)
    
    return jsonify({
        "wrong_answers": wrong_answers,
        "total_wrong": len(wrong_answers),
        "username": username
    })


@app.route('/get_user_stats')
def get_user_stats_route():
    """Get comprehensive user statistics including wrong answers"""
    global username
    
    # Try to get username from request args if not set globally
    request_username = request.args.get('username')
    if request_username:
        username = request_username.strip()
    
    return jsonify(get_user_stats(username, questions_data))

@app.route('/dashboard')
def dashboard():
    """Dashboard page showing dataset statistics"""
    return render_template('dashboard.html')


@app.route('/api/dashboard_stats')
def dashboard_stats():
    """Get comprehensive statistics for dashboard"""
    return jsonify(get_dashboard_stats(questions_data))


@app.route('/api/cache_stats')
def cache_stats():
    """Get video cache statistics"""
    return jsonify(get_cache_stats())


@app.route('/api/clear_cache', methods=['POST'])
def clear_cache_api():
    """Clear video cache"""
    data = request.json or {}
    dataset_name = data.get('dataset_name')
    video_type = data.get('video_type')
    
    try:
        clear_cache(dataset_name=dataset_name, video_type=video_type)
        return jsonify({
            "status": "success",
            "message": f"Cache cleared for dataset: {dataset_name or 'all'}, video_type: {video_type or 'all'}"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to clear cache: {str(e)}"
        }), 500


def make_all_cache(questions):
    """
    Generates and caches all videos (vis and e2vid) for all questions.
    """
    print("Starting to generate cache for all questions...")
    # Initialize the reconstructor once for efficiency
    reconstructor = VideoReconstructor()
    
    total_questions = len(questions)
    for i, question in enumerate(questions):
        dataset_name = question['dataset_name']
        question_id = question['question_id']
        h5_path = question['event_path']
        
        # Check if h5_path exists
        if not os.path.exists(h5_path):
            print(f"Warning: H5 file not found for question {question_id} in {dataset_name}. Skipping. Path: {h5_path}")
            continue

        print(f"Processing question {i+1}/{total_questions}: {dataset_name} - {question_id}")
        
        # Generate vis video if not cached
        vis_cache_path = get_cache_path(dataset_name, question_id, 'vis')
        if not is_video_cached(dataset_name, question_id, 'vis'):
            print(f"  Generating vis video...")
            success = h5_to_videos(h5_path, vis_output_path=vis_cache_path)
            if success:
                print(f"  Successfully generated and cached vis video.")
            else:
                print(f"  Failed to generate vis video.")
        
        # Generate e2vid video if not cached
        e2vid_cache_path = get_cache_path(dataset_name, question_id, 'e2vid')
        if not is_video_cached(dataset_name, question_id, 'e2vid'):
            print(f"  Generating e2vid video...")
            success = h5_to_videos(h5_path, e2vid_output_path=e2vid_cache_path, reconstructor=reconstructor)
            if success:
                print(f"  Successfully generated and cached e2vid video.")
            else:
                print(f"  Failed to generate e2vid video.")
    
    print("Cache generation complete.")


def load_questions_release_mode(dataset_root):
    """
    Load questions from a released dataset structure.
    - JSONs from {dataset_root}/questions/*.json
    - H5 files from {dataset_root}/h5_files/*/*.h5
    """
    print(f"Loading questions in release mode from: {dataset_root}")
    json_dir = os.path.join(dataset_root, 'questions')
    h5_root = os.path.join(dataset_root, 'h5_files')
    
    if not os.path.isdir(json_dir):
        print(f"Error: Questions directory not found at {json_dir}")
        return []

    # Find all H5 files first for quick lookup
    h5_files = glob.glob(os.path.join(h5_root, '**', '*.h5'), recursive=True)
    h5_map = {os.path.basename(f).replace('.h5', ''): f for f in h5_files}
    
    all_questions = []
    json_paths = glob.glob(os.path.join(json_dir, '*.json'))

    for json_path in json_paths:
        dataset_name = os.path.basename(json_path).replace(".json", "")
        all_questions.extend(load_questions_from_json_file(json_path, base_directory=os.path.join(h5_root, dataset_name)))
            
    print(f"Loaded {len(all_questions)} questions from {len(json_paths)} files in release mode.")
    return all_questions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Annotation System Server')
    parser.add_argument('--release', action='store_true', help='Enable release mode to load compiled datasets.')
    parser.add_argument('--dataset_root', type=str, default=None, help='Root directory of the compiled dataset (for release mode).')
    parser.add_argument('--make_cache', action='store_true', help='Generate and cache all videos for all questions and exit.')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host for the server.')
    parser.add_argument('--port', type=int, default=5000, help='Port for the server.')
    args = parser.parse_args()

    if args.release:
        if not args.dataset_root:
            print("Error: --dataset_root is required for --release mode.")
            sys.exit(1)
        questions_data = load_questions_release_mode(args.dataset_root)
    else:
        # Load all questions from the default 'questions' directory
        questions_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'questions')
        questions_data = load_questions(questions_dir)

    if not questions_data:
        print("No questions loaded. Exiting.")
        sys.exit(1)

    if args.make_cache:
        make_all_cache(questions_data)
        sys.exit(0)
    
    app.run(host=args.host, port=args.port, debug=True)