document.addEventListener('DOMContentLoaded', () => {
    const searchInput = document.getElementById('search-input');
    const searchButton = document.getElementById('search-button');
    const resultsContainer = document.getElementById('results');
    const prevPageButton = document.getElementById('prev-page');
    const nextPageButton = document.getElementById('next-page');
    const pageInfo = document.getElementById('page-info');

    let currentPage = 1;
    const resultsPerPage = 10;

    searchButton.addEventListener('click', () => performSearch(1));
    searchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            performSearch(1);
        }
    });

    prevPageButton.addEventListener('click', () => {
        if (currentPage > 1) {
            performSearch(currentPage - 1);
        }
    });

    nextPageButton.addEventListener('click', () => {
        performSearch(currentPage + 1);
    });

    function performSearch(page) {
        const query = searchInput.value.trim();
        if (query === '') return;

        currentPage = page;
        const limit = resultsPerPage;

        resultsContainer.innerHTML = '<p class="loading">Recherche en cours...</p>';

        fetch(`http://localhost:8000/search?query=${encodeURIComponent(query)}&limit=${limit}`)
            .then(response => response.json())
            .then(data => {
                displayResults(data.results);
                updatePagination(data.results.length);
            })
            .catch(error => {
                console.error('Erreur:', error);
                resultsContainer.innerHTML = '<p class="error">Une erreur est survenue lors de la recherche.</p>';
            });
    }

    function displayResults(results) {
        if (results.length === 0) {
            resultsContainer.innerHTML = '<p class="no-results">Aucun résultat trouvé.</p>';
            return;
        }

        let resultsHtml = '';
        results.forEach(result => {
            resultsHtml += `
                <div class="result-item">
                    <a href="${result.url}" class="result-title" target="_blank">${result.title}</a>
                    <div class="result-url">${result.url}</div>
                    <div class="result-score">Score combiné: ${result.combined_score.toFixed(4)}</div>
                    <div class="result-category">Catégorie: ${result.category}</div>
                </div>
            `;
        });

        resultsContainer.innerHTML = resultsHtml;
    }

    function updatePagination(resultCount) {
        pageInfo.textContent = `Page ${currentPage}`;
        prevPageButton.disabled = (currentPage === 1);
        nextPageButton.disabled = (resultCount < resultsPerPage);
    }
});