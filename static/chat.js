var model = null;  // Use the default model
var ws = null;
var position = 0;
var sessionMaxLength = 1024;

var totalElapsed, nRequests;

const sepToken = "\n\n";

function openSession() {
  ws = new WebSocket(`ws://${location.host}/api/v2/generate`);
  ws.onopen = () => {
    ws.send(JSON.stringify({type: "open_inference_session", model: model, max_length: sessionMaxLength}));
    ws.onmessage = event => {
      const response = JSON.parse(event.data);
      if (!response.ok) {
        handleFailure(response.traceback);
        return;
      }

      sendReplica();
    };
  };

  ws.onerror = _event => handleFailure(`Connection failed`);
  ws.onclose = _event => {
    if ($(".error-box").is(":hidden")) {
      handleFailure(`Connection was closed`);
    }
  };
}

function resetSession() {
  if (ws !== null && ws.readyState <= 1) {  // If readyState is "connecting" or "opened"
    ws.close();
  }
  ws = null;
  position = 0;
}

function isWaitingForInputs() {
  return $('.human-replica textarea').length >= 1;
}

function sendReplica() {
  if (isWaitingForInputs()) {
    $('.human-replica:last').text($('.human-replica:last textarea').val());
    $('.dialogue').append($(
      '<p class="ai-replica">' +
        '<span class="text">AI:</span><span class="loading-animation"></span>' +
        '<span class="speed" style="display: none;">Average speed: <span class="value"></span> sec/token</span>' +
        '<span class="suggest-join" style="display: none;">' +
          'This speed is slower than expected due to a high load. You can increase Petals capacity by ' +
          '<a target="_blank" href="https://github.com/bigscience-workshop/petals#connect-your-gpu-and-increase-petals-capacity">connecting your GPU</a>.' +
        '</span>' +
      '</p>'));
  } else {
    $('.loading-animation').show();
  }

  if (ws === null) {
    openSession();
    return;
  }

  const replicaDivs = $('.human-replica, .ai-replica .text');
  var replicas = [];
  for (var i = position; i < replicaDivs.length; i++) {
    replicas.push($(replicaDivs[i]).text());
  }
  const inputs = replicas.join(sepToken);
  position = replicaDivs.length;

  totalElapsed = 0;
  nRequests = 0;
  receiveReplica(inputs);
}

const textareaHtml = '<p class="human-replica"><textarea class="form-control" id="exampleTextarea" rows="2">Human: </textarea></p>';

function receiveReplica(inputs) {
  ws.send(JSON.stringify({
    type: "generate",
    inputs: inputs,
    max_new_tokens: 1,
    do_sample: 1,
    temperature: 0.75,
    top_p: 0.9,
    session_id: ws,
    stop_sequence: sepToken,
  }));

  var lastMessageTime = null;
  ws.onmessage = event => {
    const response = JSON.parse(event.data);
    if (!response.ok) {
      handleFailure(response.traceback);
      return;
    }

    if (lastMessageTime != null) {
      totalElapsed += performance.now() - lastMessageTime;
      nRequests++;
    }
    lastMessageTime = performance.now();

    const lastReplica = $('.ai-replica .text').last();
    const newText = lastReplica.text() + response.outputs;
    lastReplica.text(newText.replace(sepToken, ""));
    if (!response.stop) {
      if (nRequests >= 1) {
        const stepsPerSecond = totalElapsed / nRequests / 1000;
        $('.speed .value').text(stepsPerSecond.toFixed(1));
        $('.speed').show();
        if (stepsPerSecond >= 3) {
          $('.suggest-join').show();
        }
      }
    } else {
      $('.loading-animation, .speed, .suggest-join').remove();
      $('.dialogue').append($(textareaHtml));
      upgradeTextArea();
    }
  };
}

function handleFailure(message) {
  resetSession();
  if (!isWaitingForInputs()) {
    // Show the error and the retry button only if a user is waiting for the generation results
    var autoRetry = false;
    if (/Session .+ expired/.test(message)) {
      autoRetry = true;
    }
    const largerMaxLength = 2048;
    if (/Maximum length exceeded/.test(message) && sessionMaxLength < largerMaxLength) {
      sessionMaxLength = largerMaxLength;  // We gradually increase sessionMaxLength to save server resources
      autoRetry = true;
    }

    if (autoRetry) {
      retry();
    } else {
      $('.loading-animation').hide();
      $('.error-message').text(message);
      $('.error-box').show();
    }
  }
}

function retry() {
  $('.error-box').hide();
  sendReplica();
}

function upgradeTextArea() {
  const textarea = $('.human-replica textarea');
  autosize(textarea);
  textarea[0].selectionStart = textarea[0].value.length;
  textarea.focus();

  textarea.on('keypress', e => {
    if (e.which == 13 && !e.shiftKey) {
      e.preventDefault();
      sendReplica();
    }
  });
}

function resetDialogue() {
  if (!isWaitingForInputs()) {
    alert("Can't reset the dialogue while the AI is writing a response. Please refresh the page");
    return false;
  }
  if (!confirm("This will reset the dialogue. Are you sure?")) {
    return false;
  }

  $('.dialogue').html(textareaHtml);
  upgradeTextArea();

  resetSession();
  return true;
}

const animFrames = ["⌛", "🧠"];
var curFrame = 0;

function animateLoading() {
  $('.loading-animation').html(' &nbsp;' + animFrames[curFrame]);
  curFrame = (curFrame + 1) % animFrames.length;
}

$(() => {
  upgradeTextArea();

  $('.show-few-shot').click(e => {
    e.preventDefault();

    if (resetDialogue()) {
      const textarea = $('.human-replica textarea');
      textarea.val(
        'Human: A cat sat on a mat.\n\n' +
        'AI: Un gato se sentó en una estera.\n\n' +
        'Human: A brown fox jumps over the lazy dog.\n\n' +
        'AI: Un zorro marrón salta sobre el perro perezoso.\n\n' +
        'Human: Who is the president of the United States?'
      );
      textarea[0].style.height = textarea[0].scrollHeight + "px";
    }
  });
  $('.retry-link').click(e => {
    e.preventDefault();
    retry();
  });
  $('.use-bloomz').click(e => {
    e.preventDefault();

    if (resetDialogue()) {
      model = "bigscience/bloomz-petals";
      $('.use-bloomz-text').hide();
      $('.model-name')
        .html('BLOOMZ&#8209;176B')
        .attr('href', 'https://huggingface.co/bigscience/bloomz');

      const textarea = $('.human-replica textarea');
      textarea.val('Human: Write a Python code that prints prime numbers below 100.');
      textarea[0].style.height = textarea[0].scrollHeight + "px";
    }
  });

  setInterval(animateLoading, 2000);
});
