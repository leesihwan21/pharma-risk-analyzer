// PWA 설정: manifest, 아이콘, 테마 컬러를 동적으로 주입하고 서비스워커 등록
(function () {
  function addLink(rel, href, extra) {
    if (document.querySelector('link[rel="' + rel + '"][href="' + href + '"]')) return;
    var link = document.createElement('link');
    link.rel = rel;
    link.href = href;
    if (extra) {
      for (var k in extra) link.setAttribute(k, extra[k]);
    }
    document.head.appendChild(link);
  }

  function addMeta(name, content) {
    if (document.querySelector('meta[name="' + name + '"]')) return;
    var meta = document.createElement('meta');
    meta.name = name;
    meta.content = content;
    document.head.appendChild(meta);
  }

  addLink('manifest', '/static/manifest.json');
  addLink('icon', '/static/icon-192.png', { sizes: '192x192', type: 'image/png' });
  addLink('icon', '/static/icon-512.png', { sizes: '512x512', type: 'image/png' });
  addLink('apple-touch-icon', '/static/icon-192.png');
  addMeta('theme-color', '#0f172a');
  addMeta('apple-mobile-web-app-capable', 'yes');
  addMeta('apple-mobile-web-app-status-bar-style', 'black-translucent');

  if ('serviceWorker' in navigator) {
    window.addEventListener('load', function () {
      navigator.serviceWorker.register('/static/sw.js').catch(function (err) {
        console.warn('SW 등록 실패:', err);
      });
    });
  }
})();
