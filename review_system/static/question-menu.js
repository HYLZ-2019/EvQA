// Question Navigation Menu
class QuestionMenu {
    constructor() {
        this.questionGroups = {};
        this.isMenuOpen = false;
        this.userProgress = {
            answered_questions: [],
            total_questions: 0,
            completed_count: 0
        };
        this.init();
    }

    async init() {
        this.createMenuHTML();
        this.bindEvents();
        await this.loadQuestionGroups();
        await this.loadUserProgress();
    }

    createMenuHTML() {
        const menuHTML = `
            <div id="questionMenuOverlay" class="menu-overlay" style="display: none;" onclick="questionMenu.closeMenu()">
                <div class="menu-container" onclick="event.stopPropagation()">
                    <div class="menu-header">
                        <div class="menu-header-top">
                            <h3>Question Navigator</h3>
                            <button class="menu-close-btn" onclick="questionMenu.closeMenu()">&times;</button>
                        </div>
                        <div class="progress-container">
                            <span class="progress-text" id="progressText">0 / 0</span>
                            <div class="progress-bar-container">
                                <div class="progress-bar-fill" id="progressBarFill" style="width: 0%;"></div>
                            </div>
                            <span class="progress-percentage" id="progressPercentage">0%</span>
                        </div>
                    </div>
                    <div class="menu-content" id="menuContent">
                        <div class="loading-menu">Loading questions...</div>
                    </div>
                </div>
            </div>
            <button id="menuToggleBtn" class="menu-toggle-btn" onclick="questionMenu.toggleMenu()" title="Browse Questions">
                <span class="menu-icon">☰</span>
            </button>
        `;
        
        document.body.insertAdjacentHTML('beforeend', menuHTML);
    }

    bindEvents() {
        // Close menu with Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.isMenuOpen) {
                this.closeMenu();
            }
        });
    }

    async loadQuestionGroups() {
        try {
            const response = await fetch('/get_question_groups');
            this.questionGroups = await response.json();
            this.renderMenu();
        } catch (error) {
            console.error('Failed to load question groups:', error);
            document.getElementById('menuContent').innerHTML = 
                '<div class="error-menu">Failed to load questions</div>';
        }
    }

    async loadUserProgress() {
        try {
            const response = await fetch('/get_user_progress');
            this.userProgress = await response.json();
            this.updateProgressDisplay();
            this.renderMenu(); // Re-render to show answered/unanswered status
        } catch (error) {
            console.error('Failed to load user progress:', error);
        }
    }

    updateProgressDisplay() {
        const progressText = document.getElementById('progressText');
        const progressBarFill = document.getElementById('progressBarFill');
        const progressPercentage = document.getElementById('progressPercentage');
        
        const completed = this.userProgress.completed_count || 0;
        const total = this.userProgress.total_questions || 0;
        const percentage = total > 0 ? Math.round((completed / total) * 100) : 0;
        
        if (progressText) progressText.textContent = `${completed} / ${total}`;
        if (progressBarFill) progressBarFill.style.width = `${percentage}%`;
        if (progressPercentage) progressPercentage.textContent = `${percentage}%`;
    }

    renderMenu() {
        const menuContent = document.getElementById('menuContent');
        let html = '';

        for (const [groupName, group] of Object.entries(this.questionGroups)) {
            html += `
                <div class="question-group">
                    <div class="group-header" onclick="questionMenu.toggleGroup('${groupName}')">
                        <span class="group-name">${group.name}</span>
                        <span class="group-count">(${group.questions.length})</span>
                        <span class="group-arrow">▼</span>
                    </div>
                    <div class="group-description">${group.description}</div>
                    <div class="group-questions" id="group-${groupName}">
            `;

            group.questions.forEach((question, index) => {
                const isActive = question.index === currentIndex;
                const isAnswered = this.userProgress.answered_questions.includes(question.question_id);
                const statusClass = isAnswered ? 'answered' : 'unanswered';
                const statusIcon = isAnswered ? '✓' : '';
                
                html += `
                    <div class="question-item ${isActive ? 'active' : ''} ${statusClass}" 
                         onclick="questionMenu.selectQuestion(${question.index})"
                         data-index="${question.index}">
                        <div class="question-number">
                            ${question.index + 1}
                            ${statusIcon ? `<span class="question-status-icon ${statusClass}">${statusIcon}</span>` : ''}
                        </div>
                        <div class="question-info">
                            <div class="question-title">${question.question}</div>
                            <div class="question-meta">
                                ID: ${question.question_id} | 
                                Type: ${question.question_type || 'N/A'} |
                                Duration: ${question.duration || 'N/A'}s
                                ${isAnswered ? ' | ✅ Answered' : ' | ⏳ Pending'}
                            </div>
                        </div>
                    </div>
                `;
            });

            html += `
                    </div>
                </div>
            `;
        }

        menuContent.innerHTML = html;
    }

    toggleMenu() {
        if (this.isMenuOpen) {
            this.closeMenu();
        } else {
            this.openMenu();
        }
    }

    openMenu() {
        document.getElementById('questionMenuOverlay').style.display = 'flex';
        this.isMenuOpen = true;
        // Refresh progress and menu to show current status
        this.loadUserProgress().then(() => {
            this.updateActiveQuestion();
        });
    }

    closeMenu() {
        document.getElementById('questionMenuOverlay').style.display = 'none';
        this.isMenuOpen = false;
    }

    toggleGroup(groupName) {
        const groupElement = document.getElementById(`group-${groupName}`);
        const arrow = groupElement.parentElement.querySelector('.group-arrow');
        
        if (groupElement.style.display === 'none') {
            groupElement.style.display = 'block';
            arrow.textContent = '▼';
        } else {
            groupElement.style.display = 'none';
            arrow.textContent = '▶';
        }
    }

    async selectQuestion(index) {
        try {
            const response = await fetch('/set_question', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ index: index })
            });

            if (response.ok) {
                // Update global current index
                currentIndex = index;
                
                // Load the new question
                await loadQuestion();
                
                // Update menu to show new active question
                this.updateActiveQuestion();
                
                // Close menu
                this.closeMenu();
            } else {
                console.error('Failed to set question');
            }
        } catch (error) {
            console.error('Error selecting question:', error);
        }
    }

    updateActiveQuestion() {
        // Remove active class from all questions
        document.querySelectorAll('.question-item').forEach(item => {
            item.classList.remove('active');
        });

        // Add active class to current question
        const activeItem = document.querySelector(`[data-index="${currentIndex}"]`);
        if (activeItem) {
            activeItem.classList.add('active');
            // Scroll to active item
            activeItem.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
        
        // Update progress display
        this.updateProgressDisplay();
    }

    // Method to refresh progress after answering a question
    async refreshProgress() {
        await this.loadUserProgress();
        this.updateActiveQuestion();
    }
}

// Initialize the menu when DOM is loaded
let questionMenu;
window.addEventListener('DOMContentLoaded', function() {
    questionMenu = new QuestionMenu();
    window.questionMenu = questionMenu; // Make it globally accessible
});
