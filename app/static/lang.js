/**
 * Pharma Risk Analyzer - 한/영 언어 전환 모듈
 * 사용법: HTML 요소에 data-ko="한국어" data-en="English" 속성 추가
 */

const LANG_KEY = 'pra_lang';

function getLang() {
    return localStorage.getItem(LANG_KEY) || 'ko';
}

function setLang(lang) {
    localStorage.setItem(LANG_KEY, lang);
    applyLang(lang);
    updateLangBtn(lang);
}

function applyLang(lang) {
    document.querySelectorAll('[data-ko]').forEach(el => {
        const text = lang === 'ko' ? el.getAttribute('data-ko') : el.getAttribute('data-en');
        if (text) {
            if (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') {
                el.placeholder = text;
            } else {
                el.textContent = text;
            }
        }
    });
    // html lang 속성 업데이트
    document.documentElement.lang = lang === 'ko' ? 'ko' : 'en';
}

function updateLangBtn(lang) {
    const btn = document.getElementById('lang-btn');
    if (btn) {
        btn.textContent = lang === 'ko' ? '🇺🇸 EN' : '🇰🇷 KO';
        btn.title = lang === 'ko' ? 'Switch to English' : '한국어로 전환';
    }
}

function toggleLang() {
    const current = getLang();
    setLang(current === 'ko' ? 'en' : 'ko');
}

// 페이지 로드 시 자동 적용
document.addEventListener('DOMContentLoaded', () => {
    const lang = getLang();
    applyLang(lang);
    updateLangBtn(lang);
});
