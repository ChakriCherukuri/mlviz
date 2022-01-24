$(document).ready(function() {
  var binderUrl = null;
  var token = null;

  var buildUrl = BINDERHUB_HOST + '/build/gh/bqplot/bqplot-gallery/main';
  var evtSource = new EventSource(buildUrl);
  evtSource.onmessage = function(event) {
    var data = JSON.parse(event.data);
    $('#loader_text').html(data.phase);
    console.log(data.message);
    if (data.phase === 'ready') {
      evtSource.close();
      binderUrl = data.url;
      token = data.token;

      $('.launch-item.disabled').removeClass("disabled");
      $('.spinner-container').remove();
    }
  };

  $('.launch-item').click(function(event) {
    if (binderUrl !== null) {
      var url = $(event.currentTarget).data('example-url');

      var voilaUrl = binderUrl + 'voila/render/' + url + '?token=' + token;
      window.open(voilaUrl, '_blank').focus();
    }
  });
});
