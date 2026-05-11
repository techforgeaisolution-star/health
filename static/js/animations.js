document.addEventListener('DOMContentLoaded', function() {
  const revealElements = document.querySelectorAll('.reveal');

  const observerOptions = {
    root: null,
    rootMargin: '0px',
    threshold: 0.1
  };

  const revealObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        revealObserver.unobserve(entry.target);
      }
    });
  }, observerOptions);

  revealElements.forEach(el => revealObserver.observe(el));

  const staggerItems = document.querySelectorAll('.stagger-item');
  staggerItems.forEach((item, index) => {
    setTimeout(() => {
      item.classList.add('visible');
    }, index * 100);
  });

  const cards = document.querySelectorAll('.disease-card, .service-card, .trust-card');
  cards.forEach(card => {
    card.addEventListener('mouseenter', function(e) {
      const rect = card.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      card.style.setProperty('--mouse-x', x + 'px');
      card.style.setProperty('--mouse-y', y + 'px');
    });
  });

  const links = document.querySelectorAll('a, button');
  links.forEach(link => {
    link.addEventListener('focus', function() {
      this.style.outline = '2px solid #4f6ef7';
      this.style.outlineOffset = '2px';
    });
    link.addEventListener('blur', function() {
      this.style.outline = '';
      this.style.outlineOffset = '';
    });
  });

  let lastScrollY = window.scrollY;
  const nav = document.querySelector('.site-nav');

  window.addEventListener('scroll', () => {
    const currentScrollY = window.scrollY;

    if (currentScrollY > 100) {
      nav.style.background = 'rgba(6, 14, 36, 0.95)';
      nav.style.backdropFilter = 'blur(20px)';
    } else {
      nav.style.background = '';
      nav.style.backdropFilter = '';
    }

    lastScrollY = currentScrollY;
  }, { passive: true });

  const buttons = document.querySelectorAll('.btn-cta, .btn-predict');
  buttons.forEach(btn => {
    btn.addEventListener('click', function(e) {
      const ripple = document.createElement('span');
      ripple.classList.add('ripple');

      const rect = btn.getBoundingClientRect();
      const size = Math.max(rect.width, rect.height);

      ripple.style.width = ripple.style.height = size + 'px';
      ripple.style.left = e.clientX - rect.left - size / 2 + 'px';
      ripple.style.top = e.clientY - rect.top - size / 2 + 'px';

      btn.appendChild(ripple);

      setTimeout(() => ripple.remove(), 600);
    });
  });

  if ('scrollRestoration' in history) {
    history.scrollRestoration = 'manual';
  }

  window.addEventListener('load', () => {
    window.scrollTo(0, 0);
  });

  const navToggle = document.getElementById('nav-toggle');
  if (navToggle) {
    document.addEventListener('touchstart', function() {}, { passive: true });
  }
});