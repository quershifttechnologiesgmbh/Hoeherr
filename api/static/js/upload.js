// Drag & drop upload handling
document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const uploadBtn = document.getElementById('uploadBtn');
    const form = document.getElementById('uploadForm');

    if (!dropZone) return;

    ['dragenter', 'dragover'].forEach(evt => {
        dropZone.addEventListener(evt, (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(evt => {
        dropZone.addEventListener(evt, (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            updateFileName(files[0].name);
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            updateFileName(fileInput.files[0].name);
        }
    });

    form.addEventListener('submit', () => {
        uploadBtn.textContent = 'Wird hochgeladen...';
        uploadBtn.disabled = true;
    });

    function updateFileName(name) {
        const uploadText = dropZone.querySelector('.upload-text');
        uploadText.textContent = name;
        dropZone.querySelector('.upload-hint').textContent = 'Bereit zum Hochladen';
    }
});
