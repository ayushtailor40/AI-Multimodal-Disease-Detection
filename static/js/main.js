/* ==========================================================================
   AI MULTIMODAL DISEASE DETECTION - CLIENT-SIDE JS
   Interactive File Upload, Instant Preview, Form Validation & Tabs
   ========================================================================== */

document.addEventListener('DOMContentLoaded', function () {
  // 1. Mobile Sidebar Toggle
  const mobileToggle = document.getElementById('mobileMenuBtn');
  const sidebar = document.getElementById('appSidebar');

  if (mobileToggle && sidebar) {
    mobileToggle.addEventListener('click', function () {
      sidebar.classList.toggle('open');
    });

    // Close when clicking outside on mobile
    document.addEventListener('click', function (e) {
      if (sidebar.classList.contains('open') && !sidebar.contains(e.target) && !mobileToggle.contains(e.target)) {
        sidebar.classList.remove('open');
      }
    });
  }

  // 2. Drag & Drop File Upload with Instant Preview
  const dropzones = document.querySelectorAll('.upload-area');

  dropzones.forEach(function (dropzone) {
    const input = dropzone.querySelector('input[type="file"]');
    const container = dropzone.closest('form') || dropzone.parentElement;
    const previewContainer = container.querySelector('.preview-container');
    const previewThumbnail = container.querySelector('.preview-thumbnail');
    const previewName = container.querySelector('.preview-name');
    const previewSize = container.querySelector('.preview-size');
    const removeBtn = container.querySelector('.preview-remove');

    if (!input) return;

    ['dragenter', 'dragover'].forEach(eventName => {
      dropzone.addEventListener(eventName, function (e) {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.add('dragover');
      }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
      dropzone.addEventListener(eventName, function (e) {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.remove('dragover');
      }, false);
    });

    dropzone.addEventListener('drop', function (e) {
      const dt = e.dataTransfer;
      const files = dt.files;
      if (files.length > 0) {
        input.files = files;
        handleFileSelection(files[0]);
      }
    });

    input.addEventListener('change', function () {
      if (input.files.length > 0) {
        handleFileSelection(input.files[0]);
      }
    });

    function handleFileSelection(file) {
      const allowedExts = ['png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'];
      const fileExt = file.name.split('.').pop().toLowerCase();
      
      if (!allowedExts.includes(fileExt)) {
        alert('Invalid file format. Please upload a valid medical image (JPG, PNG, JPEG, BMP, or TIFF).');
        input.value = '';
        if (previewContainer) previewContainer.style.display = 'none';
        return;
      }

      const maxSize = 10 * 1024 * 1024; // 10 MB
      if (file.size > maxSize) {
        alert('File size exceeds 10 MB limit. Please select a smaller image.');
        input.value = '';
        if (previewContainer) previewContainer.style.display = 'none';
        return;
      }

      if (previewContainer && previewThumbnail) {
        const reader = new FileReader();
        reader.onload = function (e) {
          previewThumbnail.src = e.target.result;
          if (previewName) previewName.textContent = file.name;
          if (previewSize) previewSize.textContent = (file.size / (1024 * 1024)).toFixed(2) + ' MB';
          previewContainer.style.display = 'flex';
          dropzone.style.display = 'none';
        };
        reader.readAsDataURL(file);
      }
    }

    if (removeBtn) {
      removeBtn.addEventListener('click', function () {
        input.value = '';
        if (previewContainer) previewContainer.style.display = 'none';
        dropzone.style.display = 'block';
      });
    }
  });

  // 3. Form Submission Loading State
  const forms = document.querySelectorAll('form[enctype="multipart/form-data"]');
  forms.forEach(function (form) {
    form.addEventListener('submit', function (e) {
      const submitBtn = form.querySelector('.btn-submit');
      if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<span class="spinner"></span> Running Deep Learning Inference...';
      }
    });
  });

  // 4. Quick Demo Fill for Pneumonia Form
  window.fillPneumoniaDemo = function (profile) {
    const symptomsField = document.getElementById('symptoms');
    const ageField = document.getElementById('age');
    const tempField = document.getElementById('temperature');
    const hrField = document.getElementById('heart_rate');
    const rrField = document.getElementById('respiratory_rate');
    const spo2Field = document.getElementById('spo2');
    const coughField = document.getElementById('cough');
    const breathField = document.getElementById('breathing_difficulty');

    if (profile === 'pneumonia') {
      if (symptomsField) symptomsField.value = "High persistent fever, productive cough with yellow-green phlegm, sharp pleuritic chest pain on inhalation, chills and dyspnea.";
      if (ageField) ageField.value = "58";
      if (tempField) tempField.value = "38.9";
      if (hrField) hrField.value = "104";
      if (rrField) rrField.value = "28";
      if (spo2Field) spo2Field.value = "91";
      if (coughField) coughField.value = "1";
      if (breathField) breathField.value = "1";
    } else if (profile === 'normal') {
      if (symptomsField) symptomsField.value = "Patient presents for routine preventive checkup. Mild seasonal allergy symptoms, occasional dry throat, no chest pain or breathing discomfort.";
      if (ageField) ageField.value = "34";
      if (tempField) tempField.value = "36.6";
      if (hrField) hrField.value = "72";
      if (rrField) rrField.value = "15";
      if (spo2Field) spo2Field.value = "99";
      if (coughField) coughField.value = "0";
      if (breathField) breathField.value = "0";
    }
  };

  // 5. Tab Switching for Evaluation Page
  const tabButtons = document.querySelectorAll('.tab-btn');
  tabButtons.forEach(function (button) {
    button.addEventListener('click', function () {
      const targetId = button.getAttribute('data-tab');

      tabButtons.forEach(btn => btn.classList.remove('active'));
      button.classList.add('active');

      document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
      });

      const targetContent = document.getElementById(targetId);
      if (targetContent) {
        targetContent.classList.add('active');
      }
    });
  });
});
