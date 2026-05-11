/**
 * DiseasePredict — PDF Report Download
 * Uses html2canvas + jsPDF loaded from CDN
 */
function downloadReport(disease) {
  const btn = document.getElementById('downloadBtn');
  if (!btn) return;

  btn.classList.add('loading');
  btn.querySelector('span') && (btn.querySelector('span').textContent = 'Generating…');

  // Show report header during capture
  const target = document.getElementById('reportTarget');
  target.classList.add('pdf-capture');

  // Small delay so styles apply
  setTimeout(() => {
    html2canvas(target, {
      scale: 2,
      useCORS: true,
      backgroundColor: '#ffffff',
      logging: false,
      onclone: (doc) => {
        // Make sure fonts are loaded in clone
        doc.querySelectorAll('.btn-download, .result-actions').forEach(el => {
          el.style.display = 'none';
        });
      }
    }).then(canvas => {
      const imgData = canvas.toDataURL('image/png');
      const { jsPDF } = window.jspdf;
      const pdf = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' });

      const pageW = 210;
      const pageH = 297;
      const margin = 10;
      const usableW = pageW - margin * 2;
      const imgH = (canvas.height * usableW) / canvas.width;

      // If content is taller than page, split into pages
      let yOffset = 0;
      while (yOffset < imgH) {
        if (yOffset > 0) pdf.addPage();
        pdf.addImage(imgData, 'PNG', margin, margin - yOffset, usableW, imgH);
        yOffset += pageH - margin * 2;
      }

      const now = new Date();
      const stamp = `${now.getFullYear()}${String(now.getMonth()+1).padStart(2,'0')}${String(now.getDate()).padStart(2,'0')}`;
      pdf.save(`DiseasePredict_${disease.replace(/\s+/g,'_')}_Report_${stamp}.pdf`);

      target.classList.remove('pdf-capture');
      btn.classList.remove('loading');
      const sp = btn.querySelector('span');
      if (sp) sp.textContent = 'Download Report';
    }).catch(err => {
      console.error('PDF generation failed:', err);
      target.classList.remove('pdf-capture');
      btn.classList.remove('loading');
      alert('Could not generate PDF. Please try again.');
    });
  }, 100);
}
