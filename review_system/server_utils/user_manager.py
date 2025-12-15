"""
User management module for the annotation system.
Handles user progress tracking, logging, and answer analysis.
"""

import json
from datetime import datetime
from pathlib import Path
import os
import fcntl
import tempfile
import shutil

def get_logs_dir():
    """Get the absolute path to the logs directory"""
    this_file_path = os.path.abspath(__file__)
    log_dir_path = os.path.join(os.path.dirname(os.path.dirname(this_file_path)), "logs")
    return Path(log_dir_path)

def load_user_progress(username):
    """Load user's answered questions from their log file with improved error handling"""
    log_dir = get_logs_dir()
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"{username}.json"
    backup_file = log_dir / f"{username}.json.backup"
    
    if log_file.exists():
        try:
            with open(log_file, 'r') as f:
                logs = json.load(f)
            # Validate the data structure
            if isinstance(logs, list):
                return logs
            else:
                print(f"Invalid log file format for {username}, expected list but got {type(logs)}")
                return []
        except json.JSONDecodeError as e:
            print(f"JSON decode error loading user progress for {username}: {e}")
            
            # Try to restore from backup
            if backup_file.exists():
                try:
                    print(f"Attempting to restore from backup for {username}")
                    with open(backup_file, 'r') as f:
                        logs = json.load(f)
                    if isinstance(logs, list):
                        # Restore the main file from backup
                        shutil.copy2(backup_file, log_file)
                        print(f"Successfully restored from backup for {username}")
                        return logs
                    else:
                        print(f"Backup file also has invalid format for {username}")
                        return []
                except Exception as backup_error:
                    print(f"Failed to restore from backup for {username}: {backup_error}")
                    return []
            else:
                print(f"No backup file available for {username}")
                return []
        except Exception as e:
            print(f"Unexpected error loading user progress for {username}: {e}")
            return []
    else:
        return []


def get_answered_question_ids(username, questions_data=None):
    """Get set of question IDs that user has already answered, filtered by current question set if provided"""
    logs = load_user_progress(username)
    answered_ids = set(log['question_id'] for log in logs)
    
    # If questions_data is provided, filter to only include IDs from current question set
    if questions_data is not None:
        current_question_ids = set(q['question_id'] for q in questions_data)
        answered_ids = answered_ids.intersection(current_question_ids)
    
    return answered_ids


def get_user_wrong_answers(username, questions_data):
    """Get list of questions that user answered incorrectly (taking latest answer for each question)"""
    if not username:
        return []
    
    logs = load_user_progress(username)
    if not logs:
        return []
    
    # Group logs by question_id and keep only the latest answer for each question
    latest_answers = {}
    for log in logs:
        question_id = log['question_id']
        timestamp = log['timestamp']
        if question_id not in latest_answers or timestamp > latest_answers[question_id]['timestamp']:
            latest_answers[question_id] = log
    
    # Find questions where the user's latest answer is wrong
    wrong_questions = []
    for question_id, log in latest_answers.items():
        user_answer = log['answer']
        
        # Find the corresponding question data
        question_data = None
        question_index = None
        for i, q in enumerate(questions_data):
            if q['question_id'] == question_id:
                question_data = q
                question_index = i
                break
        
        if question_data:
            correct_answer = question_data['answer']
            if user_answer != correct_answer:
                wrong_questions.append({
                    'question_id': question_id,
                    'unique_id': question_data['unique_id'],
                    'question': question_data['question'],
                    'question_cn': question_data.get('question_cn', ''),
                    'user_answer': user_answer,
                    'correct_answer': correct_answer,
                    'correct_answer_cn': question_data.get('answer_cn', ''),
                    'dataset_name': question_data['dataset_name'],
                    'question_index': question_index,
                    'timestamp': log['timestamp']
                })
    
    # Sort by timestamp (most recent first)
    wrong_questions.sort(key=lambda x: x['timestamp'], reverse=True)
    return wrong_questions


def get_next_unanswered_question_index(username, questions_data):
    """Get the index of the next unanswered question for the user"""
    if not username:
        return 0
    
    answered_ids = get_answered_question_ids(username, questions_data)
    
    # Find the first question that hasn't been answered
    for i, question in enumerate(questions_data):
        if question['question_id'] not in answered_ids:
            return i
    
    # If all questions are answered, return the last index
    return len(questions_data) - 1 if questions_data else 0


def save_log(username, question_id, answer):
    """Save user response to log file with file locking to prevent data loss"""
    log_dir = get_logs_dir()
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"{username}.json"
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "question_id": question_id,
        "answer": answer
    }
    
    # Create backup filename
    backup_file = log_dir / f"{username}.json.backup"
    
    try:
        # Load existing logs
        logs = load_user_progress(username)
        
        # Add new entry
        logs.append(log_entry)
        
        # Write to temporary file first
        temp_file = log_dir / f"{username}.json.tmp"
        
        with open(temp_file, 'w') as f:
            # Use file locking to prevent concurrent writes
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            json.dump(logs, f, indent=2)
            f.flush()
            os.fsync(f.fileno())  # Force write to disk
        
        # Create backup of current file if it exists
        if log_file.exists():
            shutil.copy2(log_file, backup_file)
        
        # Atomically replace the original file
        shutil.move(temp_file, log_file)
        
        print(f"Successfully saved log for user {username}: {question_id} -> {answer}")
        
    except Exception as e:
        print(f"Error saving log for user {username}: {e}")
        
        # Try to restore from backup if main file is corrupted
        if backup_file.exists() and (not log_file.exists() or os.path.getsize(log_file) == 0):
            try:
                shutil.copy2(backup_file, log_file)
                print(f"Restored log file from backup for user {username}")
                # Retry saving after restoration
                logs = load_user_progress(username)
                logs.append(log_entry)
                with open(log_file, 'w') as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    json.dump(logs, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                print(f"Successfully saved log after restoration for user {username}")
            except Exception as restore_error:
                print(f"Failed to restore and save log for user {username}: {restore_error}")
                raise restore_error
        else:
            raise e
    
    finally:
        # Clean up temporary file if it still exists
        temp_file = log_dir / f"{username}.json.tmp"
        if temp_file.exists():
            try:
                temp_file.unlink()
            except:
                pass


def check_all_completed(username, questions_data):
    """Check if user has completed all questions in the current question set"""
    answered_ids = get_answered_question_ids(username, questions_data)
    all_question_ids = set(q['question_id'] for q in questions_data)
    return answered_ids >= all_question_ids


def get_user_stats(username, questions_data):
    """Get comprehensive user statistics including wrong answers for current question set"""
    if not username:
        return {
            "error": "Username is required",
            "total_questions": len(questions_data)
        }
    
    answered_ids = get_answered_question_ids(username, questions_data)
    wrong_answers = get_user_wrong_answers(username, questions_data)
    all_completed = check_all_completed(username, questions_data)
    
    # Calculate accuracy
    total_answered = len(answered_ids)
    total_wrong = len(wrong_answers)
    total_correct = total_answered - total_wrong
    accuracy = (total_correct / total_answered * 100) if total_answered > 0 else 0
    
    return {
        "username": username,
        "total_questions": len(questions_data),
        "answered_count": total_answered,
        "correct_count": total_correct,
        "wrong_count": total_wrong,
        "accuracy": round(accuracy, 2),
        "all_completed": all_completed,
        "answered_questions": list(answered_ids),
        "wrong_answers": wrong_answers
    }


def get_user_progress(username, questions_data):
    """Get user's progress - which questions they have answered in the current question set"""
    if not username:
        return {
            "answered_questions": [], 
            "total_questions": len(questions_data), 
            "completed_count": 0, 
            "all_completed": False,
            "wrong_answers": [],
            "total_wrong": 0
        }
    
    answered_ids = get_answered_question_ids(username, questions_data)
    wrong_answers = get_user_wrong_answers(username, questions_data)
    
    return {
        "answered_questions": list(answered_ids),
        "total_questions": len(questions_data),
        "completed_count": len(answered_ids),
        "all_completed": check_all_completed(username, questions_data),
        "wrong_answers": wrong_answers,
        "total_wrong": len(wrong_answers)
    }
