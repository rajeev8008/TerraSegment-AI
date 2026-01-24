// Tab Navigation
const tabBtns = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');

tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        const tabName = btn.getAttribute('data-tab');
        
        // Remove active class from all
        tabBtns.forEach(b => b.classList.remove('active'));
        tabContents.forEach(c => c.classList.remove('active'));
        
        // Add active class to clicked
        btn.classList.add('active');
        document.getElementById(tabName).classList.add('active');
    });
});

// ===== AI MODE FUNCTIONALITY =====
const uploadBox = document.getElementById('uploadBox');
const imageInput = document.getElementById('imageInput');
const aiResults = document.getElementById('aiResults');
const aiLoading = document.getElementById('aiLoading');
const aiError = document.getElementById('aiError');
const uploadAgainBtn = document.getElementById('uploadAgainBtn');

// Upload box click
uploadBox.addEventListener('click', () => imageInput.click());

// Drag and drop
uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.style.borderColor = '#6366f1';
    uploadBox.style.background = 'rgba(99, 102, 241, 0.15)';
});

uploadBox.addEventListener('dragleave', () => {
    uploadBox.style.borderColor = 'rgba(99, 102, 241, 0.5)';
    uploadBox.style.background = 'rgba(99, 102, 241, 0.05)';
});

uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.style.borderColor = 'rgba(99, 102, 241, 0.5)';
    uploadBox.style.background = 'rgba(99, 102, 241, 0.05)';
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        imageInput.files = files;
        uploadImage();
    }
});

// File input change
imageInput.addEventListener('change', uploadImage);

async function uploadImage() {
    const file = imageInput.files[0];
    if (!file) return;
    
    const formData = new FormData();
    formData.append('file', file);
    
    uploadBox.style.display = 'none';
    aiLoading.style.display = 'block';
    aiResults.style.display = 'none';
    aiError.style.display = 'none';
    
    try {
        const response = await fetch('/api/upload-image', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Upload failed');
        }
        
        const data = await response.json();
        
        aiLoading.style.display = 'none';
        
        // Display images
        document.getElementById('originalImage').src = 'data:image/png;base64,' + data.original_image;
        document.getElementById('maskImage').src = 'data:image/png;base64,' + data.segmentation_mask;
        
        // Get percentages
        const building = data.building || 0;
        const land = data.land || 0;
        const road = data.road || 0;
        const vegetation = data.vegetation || 0;
        const water = data.water || 0;
        
        // Update analysis panel
        const aiRightPanel = document.getElementById('aiRightPanel');
        aiRightPanel.style.display = 'block';
        
        // Update breakdown bars and percentages
        document.getElementById('buildingPercent').textContent = building.toFixed(2) + '%';
        document.getElementById('landPercent').textContent = land.toFixed(2) + '%';
        document.getElementById('roadPercent').textContent = road.toFixed(2) + '%';
        document.getElementById('vegetationPercent').textContent = vegetation.toFixed(2) + '%';
        document.getElementById('waterPercent').textContent = water.toFixed(2) + '%';
        
        // Update progress bars
        document.getElementById('buildingBar').style.width = building + '%';
        document.getElementById('landBar').style.width = land + '%';
        document.getElementById('roadBar').style.width = road + '%';
        document.getElementById('vegetationBar').style.width = vegetation + '%';
        document.getElementById('waterBar').style.width = water + '%';
        
        // Determine dominant class
        const classes = {
            'Building': building,
            'Land': land,
            'Road': road,
            'Vegetation': vegetation,
            'Water': water
        };
        const dominantClass = Object.keys(classes).reduce((a, b) => classes[a] > classes[b] ? a : b);
        
        // Update summary stats
        document.getElementById('dominantClass').textContent = dominantClass;
        document.getElementById('confidence').textContent = 'High';
        
        aiResults.style.display = 'block';
        
    } catch (error) {
        aiLoading.style.display = 'none';
        aiError.style.display = 'flex';
        aiError.textContent = '⚠ ' + error.message;
    }
}

uploadAgainBtn.addEventListener('click', () => {
    uploadBox.style.display = 'block';
    aiResults.style.display = 'none';
    document.getElementById('aiRightPanel').style.display = 'none';
    imageInput.value = '';
});

// Copy statistics to clipboard
document.getElementById('copyStats').addEventListener('click', () => {
    const building = document.getElementById('buildingPercent').textContent;
    const land = document.getElementById('landPercent').textContent;
    const road = document.getElementById('roadPercent').textContent;
    const vegetation = document.getElementById('vegetationPercent').textContent;
    const water = document.getElementById('waterPercent').textContent;
    const dominant = document.getElementById('dominantClass').textContent;
    
    const stats = `Terrain Analysis Report\n\n` +
        `Building: ${building}\n` +
        `Land: ${land}\n` +
        `Road: ${road}\n` +
        `Vegetation: ${vegetation}\n` +
        `Water: ${water}\n\n` +
        `Dominant Class: ${dominant}`;
    
    navigator.clipboard.writeText(stats).then(() => {
        const btn = document.getElementById('copyStats');
        const originalText = btn.innerHTML;
        btn.innerHTML = '<i class="fas fa-check"></i> Copied!';
        setTimeout(() => {
            btn.innerHTML = originalText;
        }, 2000);
    });
});

// ===== SEARCH MODE FUNCTIONALITY =====
const waterSlider = document.getElementById('waterSlider');
const roadSlider = document.getElementById('roadSlider');
const vegetationSlider = document.getElementById('vegetationSlider');
const buildingsSlider = document.getElementById('buildingsSlider');
const landSlider = document.getElementById('landSlider');

const waterValue = document.getElementById('waterValue');
const roadValue = document.getElementById('roadValue');
const vegetationValue = document.getElementById('vegetationValue');
const buildingsValue = document.getElementById('buildingsValue');
const landValue = document.getElementById('landValue');

const totalPercentage = document.getElementById('totalPercentage');
const totalStatus = document.getElementById('totalStatus');
const errorMessage = document.getElementById('errorMessage');

const displayBtn = document.getElementById('displayBtn');
const resetBtn = document.getElementById('resetBtn');

const imageContainer = document.getElementById('imageContainer');
const imageInfo = document.getElementById('imageInfo');
const loadingSpinner = document.getElementById('loadingSpinner');

// Update slider values and total
function updateSliderValues() {
    const water = parseInt(waterSlider.value);
    const road = parseInt(roadSlider.value);
    const vegetation = parseInt(vegetationSlider.value);
    const buildings = parseInt(buildingsSlider.value);
    const land = parseInt(landSlider.value);

    waterValue.textContent = water + '%';
    roadValue.textContent = road + '%';
    vegetationValue.textContent = vegetation + '%';
    buildingsValue.textContent = buildings + '%';
    landValue.textContent = land + '%';

    const total = water + road + vegetation + buildings + land;
    totalPercentage.textContent = total + '%';

    // Update progress bars
    document.querySelector('.water-progress::after').style.width = water + '%';
    document.querySelector('.road-progress::after').style.width = road + '%';
    document.querySelector('.vegetation-progress::after').style.width = vegetation + '%';
    document.querySelector('.buildings-progress::after').style.width = buildings + '%';
    document.querySelector('.land-progress::after').style.width = land + '%';

    // Check if total exceeds 100%
    if (total > 100) {
        totalStatus.classList.add('error');
        errorMessage.style.display = 'flex';
        errorMessage.textContent = '⚠ Total percentage exceeds 100%! Current: ' + total + '%';
        displayBtn.disabled = true;
        displayBtn.style.opacity = '0.5';
        displayBtn.style.cursor = 'not-allowed';
    } else {
        totalStatus.classList.remove('error');
        errorMessage.style.display = 'none';
        displayBtn.disabled = false;
        displayBtn.style.opacity = '1';
        displayBtn.style.cursor = 'pointer';
    }
}

// Event listeners for sliders
waterSlider.addEventListener('input', updateSliderValues);
roadSlider.addEventListener('input', updateSliderValues);
vegetationSlider.addEventListener('input', updateSliderValues);
buildingsSlider.addEventListener('input', updateSliderValues);
landSlider.addEventListener('input', updateSliderValues);

// Fetch and display image
displayBtn.addEventListener('click', async () => {
    const water = parseInt(waterSlider.value);
    const road = parseInt(roadSlider.value);
    const vegetation = parseInt(vegetationSlider.value);
    const buildings = parseInt(buildingsSlider.value);
    const land = parseInt(landSlider.value);

    // Show loading spinner
    loadingSpinner.style.display = 'flex';
    imageContainer.innerHTML = '';

    try {
        const response = await fetch('/api/get-image', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                water: water,
                road: road,
                vegetation: vegetation,
                buildings: buildings,
                land: land
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Failed to fetch image');
        }

        const data = await response.json();

        // Hide loading spinner
        loadingSpinner.style.display = 'none';

        // Display image
        const img = document.createElement('img');
        img.src = 'data:image/png;base64,' + data.image;
        imageContainer.innerHTML = '';
        imageContainer.appendChild(img);

        // Display info
        document.getElementById('infoWater').textContent = data.water.toFixed(2) + '%';
        document.getElementById('infoRoad').textContent = data.road.toFixed(2) + '%';
        document.getElementById('infoVegetation').textContent = data.vegetation.toFixed(2) + '%';
        document.getElementById('infoBuildings').textContent = data.buildings.toFixed(2) + '%';
        document.getElementById('infoLand').textContent = data.land.toFixed(2) + '%';

        imageInfo.style.display = 'block';
    } catch (error) {
        loadingSpinner.style.display = 'none';
        imageContainer.innerHTML = `
            <div class="placeholder" style="color: #ef4444;">
                <i class="fas fa-exclamation-circle"></i>
                <p>Error: ${error.message}</p>
            </div>
        `;
    }
});

// Reset all values
resetBtn.addEventListener('click', () => {
    waterSlider.value = 30;
    roadSlider.value = 25;
    vegetationSlider.value = 20;
    buildingsSlider.value = 15;
    landSlider.value = 10;

    updateSliderValues();

    imageContainer.innerHTML = `
        <div class="placeholder">
            <i class="fas fa-image"></i>
            <p>Select percentages and click "Find Image"</p>
        </div>
    `;
    imageInfo.style.display = 'none';
    loadingSpinner.style.display = 'none';
    errorMessage.style.display = 'none';
});

// Initialize
updateSliderValues();
