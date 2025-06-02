// Tab functionality
function openTab(evt, tabName) {
    var i, tabcontent, tablinks;
    
    // Hide all tab content
    tabcontent = document.getElementsByClassName("tab-content");
    for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].classList.remove("active");
    }
    
    // Remove active class from all tab buttons
    tablinks = document.getElementsByClassName("tab-button");
    for (i = 0; i < tablinks.length; i++) {
        tablinks[i].classList.remove("active");
    }
    
    // Show the selected tab content and mark button as active
    document.getElementById(tabName).classList.add("active");
    evt.currentTarget.classList.add("active");
}

// Parameter update functions
function updateTransactionValue(value) {
    document.getElementById('transaction-value').textContent = value;
}

function updateSupportValue(value) {
    document.getElementById('support-value').textContent = parseFloat(value).toFixed(2);
}

function updateConfidenceValue(value) {
    document.getElementById('confidence-value').textContent = parseFloat(value).toFixed(2);
}

function updateLeafSizeValue(value) {
    document.getElementById('leaf-size-value').textContent = value;
}

function updateDepthValue(value) {
    document.getElementById('depth-value').textContent = value;
}

// Toggle HashTree specific options
function toggleHashTreeOptions() {
    const algorithmSelect = document.getElementById('algorithm-select');
    const hashTreeOptions = document.getElementById('hashtree-options');
    
    if (algorithmSelect.value === 'hashtree_apriori') {
        hashTreeOptions.style.display = 'block';
    } else {
        hashTreeOptions.style.display = 'none';
    }
}

// Run algorithm function
async function runAlgorithm() {
    const algorithm = document.getElementById('algorithm-select').value;
    const numTransactions = parseInt(document.getElementById('num-transactions').value);
    const minSupport = parseFloat(document.getElementById('min-support').value);
    const minConfidence = parseFloat(document.getElementById('min-confidence').value);
    
    // Prepare request data
    const requestData = {
        algorithm: algorithm,
        num_transactions: numTransactions,
        min_support: minSupport,
        min_confidence: minConfidence
    };
    
    // Add HashTree specific parameters if needed
    if (algorithm === 'hashtree_apriori') {
        requestData.max_leaf_size = parseInt(document.getElementById('max-leaf-size').value);
        requestData.max_depth = parseInt(document.getElementById('max-depth').value);
    }
    
    // Show loading state
    showLoading();
    hideResults();
    hideError();
    
    try {
        const response = await fetch('/run_algorithm', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayResults(data);
        } else {
            showError(data.error || 'An error occurred while running the algorithm');
        }
    } catch (error) {
        showError('Network error: ' + error.message);
    } finally {
        hideLoading();
    }
}

// UI state management functions
function showLoading() {
    document.getElementById('loading').style.display = 'block';
}

function hideLoading() {
    document.getElementById('loading').style.display = 'none';
}

function showResults() {
    document.getElementById('results-header').style.display = 'flex';
    document.getElementById('results-container').style.display = 'block';
}

function hideResults() {
    document.getElementById('results-header').style.display = 'none';
    document.getElementById('results-container').style.display = 'none';
}

function showError(message) {
    document.getElementById('error-text').textContent = message;
    document.getElementById('error-message').style.display = 'flex';
}

function hideError() {
    document.getElementById('error-message').style.display = 'none';
}

// Display results function
function displayResults(data) {
    // Update stats
    document.getElementById('runtime').textContent = data.runtime;
    document.getElementById('total-itemsets').textContent = data.total_itemsets;
    
    // Clear previous results
    const tbody = document.getElementById('results-tbody');
    tbody.innerHTML = '';
    
    // Populate table with results
    data.frequent_itemsets.forEach(itemset => {
        const row = document.createElement('tr');
        
        // Size column
        const sizeCell = document.createElement('td');
        sizeCell.innerHTML = `<span class="size-badge">${itemset.size}</span>`;
        row.appendChild(sizeCell);
        
        // Itemset column
        const itemsetCell = document.createElement('td');
        const movieList = document.createElement('div');
        movieList.className = 'movie-list';
        
        itemset.itemset.forEach(movie => {
            const movieSpan = document.createElement('span');
            movieSpan.className = 'movie-item';
            movieSpan.textContent = movie;
            movieList.appendChild(movieSpan);
        });
        
        itemsetCell.appendChild(movieList);
        row.appendChild(itemsetCell);
        
        // Support column
        const supportCell = document.createElement('td');
        supportCell.innerHTML = `<span class="support-value">${itemset.support}</span>`;
        row.appendChild(supportCell);
        
        tbody.appendChild(row);
    });
    
    showResults();
}

// Initialize page
document.addEventListener('DOMContentLoaded', function() {
    // Set initial values
    updateTransactionValue(document.getElementById('num-transactions').value);
    updateSupportValue(document.getElementById('min-support').value);
    updateConfidenceValue(document.getElementById('min-confidence').value);
    updateLeafSizeValue(document.getElementById('max-leaf-size').value);
    updateDepthValue(document.getElementById('max-depth').value);
    
    // Initialize HashTree options visibility
    toggleHashTreeOptions();
});

// Add keyboard shortcuts
document.addEventListener('keydown', function(event) {
    // Enter key to run algorithm
    if (event.key === 'Enter' && event.ctrlKey) {
        runAlgorithm();
    }
    
    // Tab navigation with arrow keys
    if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
        const activeTab = document.querySelector('.tab-button.active');
        const tabs = document.querySelectorAll('.tab-button');
        const currentIndex = Array.from(tabs).indexOf(activeTab);
        
        let newIndex;
        if (event.key === 'ArrowLeft') {
            newIndex = currentIndex > 0 ? currentIndex - 1 : tabs.length - 1;
        } else {
            newIndex = currentIndex < tabs.length - 1 ? currentIndex + 1 : 0;
        }
        
        tabs[newIndex].click();
    }
});

// Add smooth scrolling to results
function scrollToResults() {
    const resultsSection = document.querySelector('.results-section');
    if (resultsSection) {
        resultsSection.scrollIntoView({ 
            behavior: 'smooth',
            block: 'start'
        });
    }
}

// Enhance the run algorithm function to scroll to results
const originalRunAlgorithm = runAlgorithm;
runAlgorithm = async function() {
    await originalRunAlgorithm();
    setTimeout(scrollToResults, 500); // Delay to ensure content is loaded
};

// Add tooltips for better UX
function addTooltips() {
    const tooltips = {
        'algorithm-select': 'Choose the frequent itemset mining algorithm to run',
        'num-transactions': 'Number of user transactions to analyze (50-1000)',
        'min-support': 'Minimum support threshold for frequent itemsets (0.01-0.5)',
        'min-confidence': 'Minimum confidence threshold for association rules (0.1-1.0)',
        'max-leaf-size': 'Maximum number of candidates in a hash tree leaf node',
        'max-depth': 'Maximum depth of the hash tree structure'
    };
    
    Object.entries(tooltips).forEach(([id, text]) => {
        const element = document.getElementById(id);
        if (element) {
            element.title = text;
        }
    });
}

// Initialize tooltips when page loads
document.addEventListener('DOMContentLoaded', addTooltips);

// Recommendation System Functions

function updateRecommendationValue(value) {
    document.getElementById('recommendation-value').textContent = value;
}

// Initialize user dropdown
function initializeUserDropdown() {
    const userSelect = document.getElementById('user-select');
    userSelect.innerHTML = '<option value="">Choose a user...</option>';
    
    // Add users 1-1000
    for (let i = 1; i <= 1000; i++) {
        const option = document.createElement('option');
        option.value = i;
        option.textContent = `User ${i}`;
        userSelect.appendChild(option);
    }
}

// Load user profile
async function loadUserProfile() {
    const userSelect = document.getElementById('user-select');
    const userId = parseInt(userSelect.value);
    
    if (!userId) {
        showRecommenderError('Please select a user first');
        return;
    }
    
    const topN = parseInt(document.getElementById('num-recommendations').value);
    
    // Show loading states
    showProfileLoading();
    hideRecommenderError();
    hideUserProfile();
    hideRecommendations();
    
    try {
        // Load user profile
        const profileResponse = await fetch('/get_user_profile', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                user_id: userId,
                top_n: topN
            })
        });
        
        const profileData = await profileResponse.json();
        
        if (profileData.success) {
            displayUserProfile(profileData.top_movies);
            
            // Load recommendations
            await loadRecommendations(userId, topN);
        } else {
            showRecommenderError(profileData.error || 'Failed to load user profile');
        }
    } catch (error) {
        showRecommenderError('Network error: ' + error.message);
    } finally {
        hideProfileLoading();
    }
}

// Load recommendations
async function loadRecommendations(userId, topK) {
    showRecommendationsLoading();
    
    try {
        const response = await fetch('/get_recommendations', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                user_id: userId,
                top_k: topK
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayRecommendations(data.recommendations);
        } else {
            showRecommenderError(data.error || 'Failed to generate recommendations');
        }
    } catch (error) {
        showRecommenderError('Network error: ' + error.message);
    } finally {
        hideRecommendationsLoading();
    }
}

// Display user profile
function displayUserProfile(movies) {
    const grid = document.getElementById('user-movies-grid');
    grid.innerHTML = '';
    
    if (!movies || movies.length === 0) {
        grid.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-film"></i>
                <h4>No movies found</h4>
                <p>This user hasn't rated any movies yet.</p>
            </div>
        `;
    } else {
        movies.forEach(movie => {
            const movieCard = createMovieCard(movie, true);
            grid.appendChild(movieCard);
        });
    }
    
    showUserProfile();
}

// Display recommendations
function displayRecommendations(recommendations) {
    // Content-based recommendations
    const contentGrid = document.getElementById('content-based-grid');
    contentGrid.innerHTML = '';
    
    if (!recommendations.content_based || recommendations.content_based.length === 0) {
        contentGrid.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-brain"></i>
                <h4>No content-based recommendations</h4>
                <p>Unable to generate content-based recommendations for this user.</p>
            </div>
        `;
    } else {
        recommendations.content_based.forEach(movie => {
            const movieCard = createMovieCard(movie, false);
            contentGrid.appendChild(movieCard);
        });
    }
    
    // Collaborative filtering recommendations
    const collabGrid = document.getElementById('collaborative-grid');
    collabGrid.innerHTML = '';
    
    if (!recommendations.collaborative || recommendations.collaborative.length === 0) {
        collabGrid.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-users"></i>
                <h4>No collaborative recommendations</h4>
                <p>Unable to generate collaborative filtering recommendations for this user.</p>
            </div>
        `;
    } else {
        recommendations.collaborative.forEach(movie => {
            const movieCard = createMovieCard(movie, false);
            collabGrid.appendChild(movieCard);
        });
    }
    
    showRecommendations();
}

// Create movie card
function createMovieCard(movie, showRating = false) {
    const card = document.createElement('div');
    card.className = 'movie-card';
    
    const title = document.createElement('div');
    title.className = 'movie-title';
    title.textContent = movie.title;
    
    const genres = document.createElement('div');
    genres.className = 'movie-genres';
    
    if (movie.genres && movie.genres !== 'Unknown') {
        const genreList = movie.genres.split('|');
        genreList.forEach(genre => {
            const genreTag = document.createElement('span');
            genreTag.className = 'genre-tag';
            genreTag.textContent = genre.trim();
            genres.appendChild(genreTag);
        });
    }
    
    card.appendChild(title);
    card.appendChild(genres);
    
    if (showRating && movie.rating) {
        const rating = document.createElement('div');
        rating.className = 'movie-rating';
        rating.innerHTML = `<i class="fas fa-star"></i> ${movie.rating}`;
        card.appendChild(rating);
    }
    
    return card;
}

// UI state management for recommender system
function showProfileLoading() {
    document.getElementById('profile-loading').style.display = 'block';
}

function hideProfileLoading() {
    document.getElementById('profile-loading').style.display = 'none';
}

function showRecommendationsLoading() {
    document.getElementById('recommendations-loading').style.display = 'block';
}

function hideRecommendationsLoading() {
    document.getElementById('recommendations-loading').style.display = 'none';
}

function showUserProfile() {
    document.getElementById('user-profile-section').style.display = 'block';
}

function hideUserProfile() {
    document.getElementById('user-profile-section').style.display = 'none';
}

function showRecommendations() {
    document.getElementById('recommendations-section').style.display = 'block';
}

function hideRecommendations() {
    document.getElementById('recommendations-section').style.display = 'none';
}

function showRecommenderError(message) {
    document.getElementById('recommender-error-text').textContent = message;
    document.getElementById('recommender-error-message').style.display = 'flex';
}

function hideRecommenderError() {
    document.getElementById('recommender-error-message').style.display = 'none';
}

// Initialize recommendation system when page loads
document.addEventListener('DOMContentLoaded', function() {
    initializeUserDropdown();
    updateRecommendationValue(document.getElementById('num-recommendations').value);
}); 