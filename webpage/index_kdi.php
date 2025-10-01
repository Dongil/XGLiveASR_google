<!-- index_kdi.php -->
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <title>실시간 음성인식 및 다중 번역 (Google STT)</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js"></script>
  <!-- [추가] QR 코드 생성을 위한 라이브러리 -->
  <script src="https://cdn.jsdelivr.net/npm/qrcodejs@1.0.0/qrcode.min.js"></script>
  <link rel="stylesheet" type="text/css" href="https://asr.xenoglobal.co.kr/css/fontstyle.css">  
  <style>
   ::-webkit-scrollbar {display: none;} 
    :root {
      --bg-color: #1a1a1a; --card-bg-color: #2c2c2c; --text-color: #e0e0e0;
      --text-color-strong: #f5f5f5; --border-color: #444; --control-bg-color: #3e3e3e;
      --control-hover-bg-color: #4a4a4a; --log-wrapper-bg-color: #222;
      --primary-color: #4a90e2; --primary-hover-color: #5aa1f2; --disabled-bg-color: #555;
      --disabled-text-color: #999; --meter-bg-color: #444; --meter-bar-color: #4caf50;
      --interim-text-color: #9e9e9e;
    }
    @media (prefers-color-scheme: light) {
      :root {
        --bg-color: #f5f5f7; --card-bg-color: #ffffff; --text-color: #333333;
        --text-color-strong: #111111; --border-color: #e0e0e0; --control-bg-color: #f1f3f5;
        --control-hover-bg-color: #e9ecef; --log-wrapper-bg-color: #f8f9fa;
        --disabled-bg-color: #e9ecef; --disabled-text-color: #aaaaaa; --meter-bg-color: #e0e0e0;
        --interim-text-color: #757575;
      }
    }
    html, body {
      background-color: var(--bg-color); color: var(--text-color);
      font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
      margin: 0; padding: 0; height: 100vh; box-sizing: border-box;
      min-height: 450px; overflow: hidden;
    }
    h3 {
      font-weight: 600; color: var(--text-color-strong); margin: 0; padding: 0; 
      border: none;
    }
    .card {
      background-color: var(--card-bg-color); border: 1px solid var(--border-color);
      border-radius: 12px; padding: 1rem; box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .row { display: flex; flex-direction: column; gap: .5rem; }
    .btn-row { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
    label { font-size: 0.95rem; color: var(--text-color); }
    select, button, textarea {
      width: 100%; padding: 0.8rem; background-color: var(--control-bg-color); color: var(--text-color);
      border: 1px solid var(--border-color); border-radius: 8px; font-size: 1rem; cursor: pointer;
      transition: background-color 0.2s ease, border-color 0.2s ease;
      box-sizing: border-box;
    }
    textarea { font-family: 'SFMono-Regular', Menlo, Consolas, 'Liberation Mono', monospace; font-size: 0.9em; resize: vertical; min-height: 200px; }
    textarea#systemLogTextarea { min-height: 100px; background-color: var(--log-wrapper-bg-color); cursor: default; }
    button { background-color: var(--primary-color); border: none; font-weight: 600; }
    button:hover:not(:disabled) { background-color: var(--primary-hover-color); }
    button:disabled { background-color: var(--disabled-bg-color); cursor: not-allowed; color: var(--disabled-text-color); }
    .meter { height: 12px; background: var(--meter-bg-color); border-radius: 6px; overflow: hidden; margin-top: 0.5rem; }
    .bar { height: 100%; width: 0%; background: var(--meter-bar-color); transition: width .08s linear; }
    .mono { font-family: 'SFMono-Regular', Menlo, Consolas, 'Liberation Mono', monospace; font-size: 2.2em; }

    .main-container {
      display: grid; grid-template-columns: 380px 1fr; gap: 1rem;
      height: calc(100vh - 2rem); position: relative; padding:1rem;
      transition: grid-template-columns 0.4s ease-in-out;
    }
    .controls-card { 
      position: relative; display: flex; flex-direction: column; 
      gap: 1.2rem; overflow-y: auto; overflow-x: hidden;
      transition: visibility 0.4s;
    }
    .main-container.sidebar-collapsed { grid-template-columns: 0 1fr; }
    .main-container.sidebar-collapsed .controls-card { visibility: hidden; }
    #sidebarToggleBtn {
      position: absolute; top: 50%; left: 410px; transform: translate(-50%, -50%); z-index: 101;
      width: 28px; height: 50px; padding: 0; border-radius: 0 8px 8px 0;
      background-color: var(--control-hover-bg-color); border: 1px solid var(--border-color); border-left: none;
      font-size: 1.2rem; display: flex; align-items: center; justify-content: center;
      cursor: pointer; transition: left 0.4s ease-in-out, background-color 0.2s ease; color: var(--text-color);
    }
    #sidebarToggleBtn:hover { background-color: var(--primary-color); }
    .main-container.sidebar-collapsed #sidebarToggleBtn { left:10px; }
    
    .output-card { display: flex; flex-direction: column; overflow: hidden; min-height: 0; min-width:300px; }
    .output-header { display: flex; justify-content: space-between; align-items: center; gap: 1rem; margin-bottom: 1rem; flex-wrap: wrap-reverse; }
    .view-options { display: flex; gap: 1rem; flex-wrap: wrap; flex-grow: 1; }
    .view-options label {
        display: flex; align-items: center; gap: 0.4rem; cursor: pointer;
        padding: 0.5rem 0.8rem; background-color: var(--control-bg-color); border-radius: 20px;
    }
    .output-grid { flex-grow: 1; display: grid; gap: 1rem; min-height: 0; }
    .log-container-wrapper { display: flex; flex-direction: column; background-color: var(--log-wrapper-bg-color); border-radius: 8px; padding: 0.8rem; overflow: hidden; min-height: 0; }
    .log-container-wrapper.hidden { display: none; }
    .log-container { flex-grow: 1; font-family: 'Noto Sans KR'; font-weight:500; overflow-y: auto; line-height: 1.6; word-break: keep-all; min-height: 0; }
	
    .log-container {word-break: auto-phrase}
    
    .log-container p { position: relative; margin: 0 0 0.7em 0; padding-left: 1.2em; }
    .log-container p::before {
      content: ''; position: absolute; left: 0; top: 0.3em;
      height: calc(100% - 0.4em); width: 3px; background-color: var(--primary-color); border-radius: 3px;
    }
    .log-container p.interim { color: var(--interim-text-color); }
    .log-container p.interim::before { background-color: var(--interim-text-color); }
    
    .log-header { display: flex; justify-content: space-between; align-items: center; gap: 0.5rem; border-bottom: 1px solid var(--border-color); padding-bottom: 0.8rem; margin-bottom: 0.8rem; }
    .download-btn {
      background: transparent; border: none; color: var(--text-color); width: 32px; height: 32px;
      padding: 0; border-radius: 50%; font-size: 1.5rem; line-height: 1;
      cursor: pointer; transition: background-color 0.2s ease, color 0.2s ease, visibility 0.2s;
      display: flex; align-items: center; justify-content: center; flex-shrink: 0;
      visibility: hidden;
    }
    .log-header.has-content .download-btn { visibility: visible; }
    .download-btn:hover { background-color: var(--control-hover-bg-color); color: var(--text-color-strong); }

    .section-header { 
      display: flex; 
      align-items: center;
      gap: 0.5rem;
    }
    .config-toggle-btn { width: 32px; height: 32px; padding: 0; flex-shrink: 0; font-size: 1.5rem; border-radius: 50%; }
    .json-config-area { display: none; flex-direction: column; gap: 0.8rem; margin-top: 1rem; }
    .json-config-area.visible { display: flex; }
    .json-config-area .btn-row { grid-template-columns: repeat(3, 1fr); }
    .json-config-area p { font-size: 0.8em; margin: 0; opacity: 0.7; text-align: center; }

    #mobileControlsToggleBtn, #closeOverlayBtn { display: none; } 

    .mobile-start-stop {
        display: none;
        gap: 1rem;
    }
    .mobile-start-stop button {
        padding: 0.5rem 1rem;
    }

	 @media (max-width: 400px), (max-height: 400px) {
		 
     #sidebarToggleBtn { display: none; } 
	 .main-container {padding:0.5rem;}
	 
	 }
   
    @media (max-width: 480px) {
        .mono {
            font-size: 1.1em;
        }
    }
  </style>
</head>
<body>
  <div class="main-container" id="mainContainer">
    <div class="card controls-card" id="controlsCard">
      <button id="closeOverlayBtn" aria-label="설정 닫기">&times;</button>
      <input type="hidden" id="wsUrl" value="wss://ai1.xenoglobal.co.kr:9500/ws">
      <section>
        <div class="section-header">
            <h3>장치 및 연결 설정</h3>
            <button id="toggleJsonConfigBtn" class="config-toggle-btn" aria-label="JSON 설정 열기">&#9881;</button>
            <!-- [추가] 학생용 QR 코드 보기 버튼 (index_vad_trans2_30_bro.php 참고) -->
            <button id="showQrBtn" class="config-toggle-btn" aria-label="학생용 QR 코드 보기" style="margin-left: auto;">&#x25A3;</button>
        </div>


        <div id="jsonConfigArea" class="json-config-area">
<!--
            <div class="btn-row">
                <button id="btnGetConfig">서버 설정 가져오기</button>
                <button id="btnApplyConfig">적용(일시)</button>
                <button id="btnSaveConfig">저장(서버에 영구)</button>
            </div>
            <label for="configTextarea">설정 (JSON)</label>
            <textarea id="configTextarea" rows="10"></textarea>
            <p>※ 적용은 현재 접속에만, 저장은 config.json에 영구 반영됩니다.</p>
//-->            
            <label for="systemLogTextarea" style="margin-top: 1rem;">시스템 로그</label>
            <textarea id="systemLogTextarea" rows="6" readonly></textarea>
            <p>※ 서버 연결 상태, 오류 등의 정보가 표시됩니다.</p>
        </div>


        <div class="row"> <label for="deviceSelect">장치 선택</label> <select id="deviceSelect"></select> </div>
        <div class="meter"><div id="bar" class="bar"></div></div>
        <div id="stat" class="mono" style="font-size: 0.8em; margin-top: 1rem; text-align: center;">-</div>
        <div class="btn-row"> <button id="btnStart">시작</button> <button id="btnStop" disabled>중지</button> </div>
      </section>
      
      <section>
        <h3>번역 설정</h3>
        <div class="row"> <label for="translationEngine">번역 엔진</label> <select id="translationEngine"> <option value="">번역 안함</option> <option value="deepl">DeepL</option> <option value="papago" selected>Papago</option> <option value="google">Google</option> </select> </div>
        <div class="row"> <label for="targetLanguage1">번역 언어 1</label> <select id="targetLanguage1"> <option value="" selected>선택 안함</option> <option value="en">English</option> <option value="ja">Japanese</option> <option value="zh">Chinese</option> <option value="es">Spanish</option> <option value="fr">French</option> <option value="de">German</option> <option value="ru">Russian</option> <option value="vi">Vietnamese</option> </select> </div>
        <div class="row"> <label for="targetLanguage2">번역 언어 2</label> <select id="targetLanguage2"> <option value="" selected>선택 안함</option> <option value="en">English</option> <option value="ja">Japanese</option> <option value="zh">Chinese</option> <option value="es">Spanish</option> <option value="fr">French</option> <option value="de">German</option> <option value="ru">Russian</option> <option value="vi">Vietnamese</option> </select> </div>
        <div class="row"> <label for="targetLanguage3">번역 언어 3</label> <select id="targetLanguage3"> <option value="" selected>선택 안함</option> <option value="en">English</option> <option value="ja">Japanese</option> <option value="zh">Chinese</option> <option value="es">Spanish</option> <option value="fr">French</option> <option value="de">German</option> <option value="ru">Russian</option> <option value="vi">Vietnamese</option> </select> </div>
      </section>

      <section>
        <h3>부가 기능</h3>
        <div class="row">
          <label for="recordAudioToggle" style="display:flex; align-items:center; justify-content:space-between; cursor:pointer;">
            <span>마이크 소리 녹음하기</span>
            <input type="checkbox" id="recordAudioToggle">
          </label>
        </div>
      </section>

	   <h3 id="messageWindowHeader">메세지창</h3>
	  <div class="output-header">
        <div class="view-options">
            <label><input type="checkbox" id="toggleOriginal" checked> 원본 (한국어)</label>
            <label><input type="checkbox" id="toggleTranslated1" checked> 번역 1</label>
            <label><input type="checkbox" id="toggleTranslated2" checked> 번역 2</label>
            <label><input type="checkbox" id="toggleTranslated3" checked> 번역 3</label>
        </div>
        
        <div style="display: none; gap: 1rem; align-items: center;">
            <div class="mobile-start-stop">
                <button id="btnStartMobile">시작</button>
                <button id="btnStopMobile" disabled>중지</button>
            </div>
            <button id="mobileControlsToggleBtn">장치 및 설정</button>
        </div>
      </div>

    </div>
    
    <button id="sidebarToggleBtn">&lt;</button>

    <div class="card output-card">
      
      <div class="output-grid" id="outputGrid">
        <div id="log-original-wrapper" class="log-container-wrapper">
          <div class="log-header">
            <h3 id="titleOriginal">메시지 원본 (한국어)</h3>
            <button class="download-btn" data-target="log-original" aria-label="메시지 원본 다운로드">&#x2193;</button>
          </div>
          <div id="log-original" class="mono log-container"></div>
        </div>
        <div id="log-translated1-wrapper" class="log-container-wrapper">
          <div class="log-header">
            <h3 id="titleTranslated1">번역 결과 1</h3>
            <button class="download-btn" data-target="log-translated1" aria-label="번역 결과 1 다운로드">&#x2193;</button>
          </div>
          <div id="log-translated1" class="mono log-container"></div>
        </div>
        <div id="log-translated2-wrapper" class="log-container-wrapper">
          <div class="log-header">
            <h3 id="titleTranslated2">번역 결과 2</h3>
            <button class="download-btn" data-target="log-translated2" aria-label="번역 결과 2 다운로드">&#x2193;</button>
          </div>
          <div id="log-translated2" class="mono log-container"></div>
        </div>
        <div id="log-translated3-wrapper" class="log-container-wrapper">
          <div class="log-header">
            <h3 id="titleTranslated3">번역 결과 3</h3>
            <button class="download-btn" data-target="log-translated3" aria-label="번역 결과 3 다운로드">&#x2193;</button>
          </div>
          <div id="log-translated3" class="mono log-container"></div>
        </div>
      </div>
    </div>
  </div>

  <!-- [추가] QR 코드 표시를 위한 모달 창 (index_vad_trans2_30_bro.php 참고) -->
  <div id="qrModal" style="display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; overflow: auto; background-color: rgba(0,0,0,0.6); align-items: center; justify-content: center;">
    <div id="qrModalContent" style="background-color: var(--card-bg-color); padding: 25px; border: 1px solid var(--border-color); border-radius: 12px; width: 90%; max-width: 450px; text-align: center; position: relative;">
      <span id="closeQrModal" style="color: var(--text-color); position: absolute; top: 10px; right: 20px; font-size: 28px; font-weight: bold; cursor: pointer;">&times;</span>
      <h3 style="margin-bottom: 1.2rem;">학생용 접속 QR 코드</h3>
      <div id="qrCodeContainer" style="padding: 1rem; background-color: white; border-radius: 8px; display: inline-block;"></div>
      <p style="margin-top: 1.2rem; font-size: 0.9em; color: var(--text-color); line-height: 1.5;">학생들은 이 QR 코드를 스캔하여<br>실시간 메시지를 확인할 수 있습니다.</p>
    </div>
  </div>

<script>
    // --- Element Selectors ---
    const mainContainer = document.getElementById('mainContainer');
    const controlsCard = document.getElementById('controlsCard');
    const sidebarToggleBtn = document.getElementById('sidebarToggleBtn');
    const mobileControlsToggleBtn = document.getElementById('mobileControlsToggleBtn');
    const closeOverlayBtn = document.getElementById('closeOverlayBtn');
    const outputGrid = document.getElementById('outputGrid');
    const wsUrlEl = document.getElementById('wsUrl');
    const btnStart = document.getElementById('btnStart');
    const btnStop = document.getElementById('btnStop');
    const btnStartMobile = document.getElementById('btnStartMobile');
    const btnStopMobile = document.getElementById('btnStopMobile');
    const deviceSelect = document.getElementById('deviceSelect');
    const translationEngineEl = document.getElementById('translationEngine');
    const toggleJsonConfigBtn = document.getElementById('toggleJsonConfigBtn');
    const jsonConfigArea = document.getElementById('jsonConfigArea');
    const configTextarea = document.getElementById('configTextarea');
    const btnGetConfig = document.getElementById('btnGetConfig');
    const btnApplyConfig = document.getElementById('btnApplyConfig');
    const btnSaveConfig = document.getElementById('btnSaveConfig');
    const systemLogTextarea = document.getElementById('systemLogTextarea');
    const recordAudioToggle = document.getElementById('recordAudioToggle');
    
    const logElements = [ document.getElementById('log-original'), document.getElementById('log-translated1'), document.getElementById('log-translated2'), document.getElementById('log-translated3') ];
    const logWrappers = [ document.getElementById('log-original-wrapper'), document.getElementById('log-translated1-wrapper'), document.getElementById('log-translated2-wrapper'), document.getElementById('log-translated3-wrapper') ];
    const toggleCheckboxes = [ document.getElementById('toggleOriginal'), document.getElementById('toggleTranslated1'), document.getElementById('toggleTranslated2'), document.getElementById('toggleTranslated3') ];
    const translationElements = [
        { select: document.getElementById('targetLanguage1'), title: document.getElementById('titleTranslated1'), baseTitle: '번역 결과 1' },
        { select: document.getElementById('targetLanguage2'), title: document.getElementById('titleTranslated2'), baseTitle: '번역 결과 2' },
        { select: document.getElementById('targetLanguage3'), title: document.getElementById('titleTranslated3'), baseTitle: '번역 결과 3' }
    ];

    // --- [신규] 언어 및 엔진 지원 데이터 정의 ---
    const ALL_LANGUAGES = [
        { code: "en", name: "English" }, 
        { code: "ja", name: "Japanese" },
        { code: "zh", name: "Chinese" }, 
        { code: "vi", name: "Vietnamese" },
        { code: "id", name: "Indonesian" }, 
        { code: "th", name: "Thai" },
        { code: "mn", name: "Mongolian" }, 
        { code: "uz", name: "Uzbek" },
        { code: "tr", name: "Turkish" }, 
        { code: "de", name: "German" },
        { code: "it", name: "Italian" }, 
        { code: "fr", name: "French" },
        { code: "es", name: "Spanish" }, 
        { code: "ru", name: "Russian" },
        { code: "pt", name: "Portuguese" }
    ];

    /* 
    const ALL_LANGUAGES = [
    { code: "en", name: "영어" },
    { code: "ja", name: "일본어" },
    { code: "zh", name: "중국어" },
    { code: "vi", name: "베트남어" },
    { code: "id", name: "인도네시아어" },
    { code: "th", name: "태국어" },
    { code: "mn", name: "몽골어" },
    { code: "uz", name: "우즈벡어" },
    { code: "tr", name: "터키어" },
    { code: "de", name: "독일어" },
    { code: "it", name: "이탈리아어" },
    { code: "fr", name: "프랑스어" },
    { code: "es", name: "스페인어" },
    { code: "ru", name: "러시아어" },
    { code: "pt", name: "포르투갈어" }
    ];
    */

    const LANGUAGE_NAMES = Object.fromEntries(ALL_LANGUAGES.map(lang => [lang.code, lang.name]));

    const SUPPORTED_LANGUAGES_BY_ENGINE = {
        'deepl': ['en', 'ja', 'zh', 'vi', 'id', 'tr', 'de', 'it', 'fr', 'es', 'ru', 'pt'],
        'papago': ['en', 'ja', 'zh', 'vi', 'id', 'th', 'de', 'it', 'fr', 'es', 'ru'],
        'google': ['en', 'ja', 'zh', 'vi', 'id', 'th', 'mn', 'uz', 'tr', 'de', 'it', 'fr', 'es', 'ru', 'pt']
    };
    // --- [신규] 정의 종료 ---

    // --- Global State ---
    let ws, ac, src, stream, worklet, rafId;
    let bytesSent = 0, seq = 0;
    
    let mediaRecorder = null;
    let recordedChunks = [];
    let recordingStarted = false;
   
    // --- [신규] 선택된 번역 언어를 앞으로 당겨 정리하는 함수 ---
    function compactLanguageSelections() {
        // 1. 현재 선택된 모든 언어 값을 수집 (중복 제거 포함)
        const selectedValues = translationElements.map(item => item.select.value);
        const validSelections = selectedValues.filter(val => val !== "");
        const uniqueSelections = [...new Set(validSelections)];

        // 2. 수집된 값을 1, 2, 3번 select에 순서대로 재할당
        translationElements.forEach((item, index) => {
            if (index < uniqueSelections.length) {
                item.select.value = uniqueSelections[index];
            } else {
                item.select.value = ""; // 남는 자리는 '선택 안함'으로
            }
        });
    }

    // --- [신규] 번역 언어 옵션 업데이트 함수 ---
    function updateLanguageOptions() {
        const selectedEngine = translationEngineEl.value;
        const supportedLangs = SUPPORTED_LANGUAGES_BY_ENGINE[selectedEngine] || [];

        translationElements.forEach(item => {
            const selectEl = item.select;
            const previouslySelectedValue = selectEl.value;
            
            // 현재 선택된 옵션들을 모두 제거 (첫 번째 '선택 안함' 옵션 제외)
            while (selectEl.options.length > 1) {
                selectEl.remove(1);
            }

            // 지원하는 언어 목록으로 새로운 옵션 추가
            if (selectedEngine) { // 엔진이 선택된 경우에만
                supportedLangs.forEach(langCode => {
                    const option = document.createElement('option');
                    option.value = langCode;
                    option.textContent = LANGUAGE_NAMES[langCode] || langCode;
                    selectEl.appendChild(option);
                });
            }

            // 이전 선택값 유지 시도
            if (supportedLangs.includes(previouslySelectedValue)) {
                selectEl.value = previouslySelectedValue;
            } else {
                selectEl.value = ""; // 지원하지 않으면 '선택 안함'으로
            }
        });
    }
    // --- [신규] 함수 종료 ---

    // --- UI/UX Helper Functions ---
    const W_BREAKPOINT = 900;
    const H_BREAKPOINT = 500;
    
    function updateWindowsAndLayout() {
        logWrappers[0].classList.toggle('hidden', !toggleCheckboxes[0].checked);
        translationElements.forEach((item, index) => {
            const langIsSelected = item.select.value !== '';
            const checkboxIsChecked = toggleCheckboxes[index + 1].checked;
            logWrappers[index + 1].classList.toggle('hidden', !(langIsSelected && checkboxIsChecked));
        });
        updateOutputGridLayout();
    }
    
    function updateOutputGridLayout() {
        const visibleCount = logWrappers.filter(wrapper => !wrapper.classList.contains('hidden')).length;
        if (visibleCount === 0) return;
        const width = window.innerWidth;
        const height = window.innerHeight;
        outputGrid.style.gridTemplateRows = '';
        outputGrid.style.gridTemplateColumns = '';
        if (width <= W_BREAKPOINT) {
            outputGrid.style.gridTemplateColumns = '1fr';
            outputGrid.style.gridTemplateRows = `repeat(${visibleCount}, 1fr)`;
        } else if (height <= H_BREAKPOINT) {
            outputGrid.style.gridTemplateRows = '1fr';
            outputGrid.style.gridTemplateColumns = `repeat(${visibleCount}, 1fr)`;
        } else {
            if (visibleCount === 1) {
                outputGrid.style.gridTemplateColumns = '1fr';
            } else {
                outputGrid.style.gridTemplateColumns = 'repeat(2, 1fr)';
            }
            const numRows = Math.ceil(visibleCount / 2);
            outputGrid.style.gridTemplateRows = `repeat(${numRows}, 1fr)`;
        }
    }

    function toggleSidebar() { mainContainer.classList.toggle('sidebar-collapsed'); sidebarToggleBtn.innerHTML = mainContainer.classList.contains('sidebar-collapsed') ? '&gt;' : '&lt;'; }
    function toggleMobileOverlay() { controlsCard.classList.toggle('overlay-visible'); }
    function updateTranslationTitles() {
        translationElements.forEach(item => {
            const selectedOption = item.select.options[item.select.selectedIndex];
            item.title.textContent = (selectedOption && selectedOption.value) ? `${item.baseTitle} - [${selectedOption.text}]` : item.baseTitle;
        });
    }
    
    // --- Download Functions ---
    function getLogContent(logContainer) {
        return Array.from(logContainer.querySelectorAll('p:not(.interim)')).map(p => p.textContent).join('\n');
    }
    function downloadFile(blob, filename) {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url; a.download = filename;
        document.body.appendChild(a); a.click();
        document.body.removeChild(a); URL.revokeObjectURL(url);
    }

	/*
    function handleDownload(event) {
        const button = event.currentTarget;
        const targetId = button.dataset.target;
        const logContainer = document.getElementById(targetId);
        const titleElement = button.parentElement.querySelector('h3');
        if (!logContainer || !titleElement) return;
        const content = getLogContent(logContainer);
        if (!content) { alert('다운로드할 내용이 없습니다.'); return; }
        const filename = titleElement.textContent.trim().replace(/[\s\[\]]/g, '_') + '.txt';
        const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
        downloadFile(blob, filename);
    }
	*/
	// --- 기존 handleDownload 함수를 삭제하고 아래 코드로 교체하세요 ---

	function handleDownload(event) {
		const button = event.currentTarget;
		
		// 1. 버튼에서 가장 가까운 '.log-container-wrapper'를 찾습니다.
		//    이 wrapper가 제목과 내용 컨테이너를 모두 감싸고 있습니다.
		const wrapper = button.closest('.log-container-wrapper');
		if (!wrapper) {
			console.error("Could not find parent wrapper for download button.");
			return;
		}

		// 2. wrapper 안에서 제목(h3)과 내용 컨테이너(.log-container)를 찾습니다.
		//    이 방식은 ID나 data-target 속성에 의존하지 않아 훨씬 안정적입니다.
		const logContainer = wrapper.querySelector('.log-container');
		const titleElement = wrapper.querySelector('h3');

		if (!logContainer || !titleElement) {
			console.error("Could not find log container or title element within wrapper.");
			return;
		}

		// 3. 내용을 가져옵니다. (getLogContent 함수는 그대로 사용)
		const content = getLogContent(logContainer);

		if (!content) {
			alert('다운로드할 내용이 없습니다.');
			return;
		}

		// 4. 파일 이름 생성 및 다운로드 (기존 로직과 동일)
		const filename = titleElement.textContent.trim().replace(/[\s\[\]]/g, '_') + '.txt';
		const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
		downloadFile(blob, filename);
	}	
    
    async function downloadAllFilesAsZip() {
        const zip = new JSZip();
        
        logWrappers.forEach((wrapper, index) => {
            if (!wrapper.classList.contains('hidden')) {
                const logContainer = logElements[index];
                const content = getLogContent(logContainer);
                if (content) {
                    const titleElement = wrapper.querySelector('h3');
                    const filename = titleElement.textContent.trim().replace(/[\s\[\]]/g, '_') + '.txt';
                    zip.file(filename, content);
                }
            }
        });

        const ts = new Date();
        const pad = (n) => String(n).padStart(2, '0');
        const tsName = `${ts.getFullYear()}${pad(ts.getMonth()+1)}${pad(ts.getDate())}_${pad(ts.getHours())}${pad(ts.getMinutes())}${pad(ts.getSeconds())}`;

        const clientRecBlob = await getClientRecordingBlob();
        if (clientRecBlob && clientRecBlob.size > 0) {
            const recExt = clientRecBlob.type.includes('ogg') ? 'ogg' : 'webm';
            zip.file(`client_recording_${tsName}.${recExt}`, clientRecBlob);
        }

        const blob = await zip.generateAsync({ type: "blob" });
        if (Object.keys(zip.files).length > 0) {
            downloadFile(blob, `logs_${tsName}.zip`);
        } else {
            alert("다운로드할 내용이 없습니다.");
        }
    }

    // --- Core Logic ---
    function clearLogs() { 
        logElements.forEach(el => {
            el.innerHTML = '';
            el.closest('.log-container-wrapper').querySelector('.log-header').classList.remove('has-content');
        });
    }
    function logSystemMessage(text) {
        const timestamp = new Date().toLocaleTimeString();
        systemLogTextarea.value += `[${timestamp}] ${text}\n`;
        systemLogTextarea.scrollTop = systemLogTextarea.scrollHeight;
    }
    function clearSystemLog() {
        systemLogTextarea.value = '';
    }
    
//var remainingText = '';	

    // [최종 수정] sttInterimHandler: 미확정 텍스트를 표시/제거하는 역할만 수행
    function sttInterimHandler(data) {
        const container = logElements[0];
        let p = container.querySelector('p.interim');

        // 서버가 빈 텍스트를 보내면 회색 문단을 완전히 제거
        if (!data.text.trim()) {
            if (p) p.remove();
            return;
        }

        // 회색 문단이 없으면 새로 생성
        if (!p) {
            p = document.createElement('p');
            p.classList.add('interim');
            container.appendChild(p);
        }
        
        // 내용 업데이트 및 스크롤
        p.textContent = data.text;
        container.scrollTop = container.scrollHeight;
    }

    // [최종 수정] sttFinalHandler: 최종 문장을 추가하는 역할만 수행
    function sttFinalHandler(data) {
        const container = logElements[0];
        const { sentence_id, text } = data;

        // 기존의 회색 문단은 interimHandler가 지워줄 것이므로 여기서는 신경쓰지 않음.
        // ID를 기준으로 문단을 찾아보고 없으면 새로 생성.
        let p = container.querySelector(`p[data-id="${sentence_id}"]`);
        if (!p) {
            p = document.createElement('p');
            p.dataset.id = sentence_id;

            // 새로 생성할 때, 기존의 interim 문단이 있다면 그 앞에 삽입하여 순서를 맞춤
            const interimP = container.querySelector('p.interim');
            if (interimP) {
                container.insertBefore(p, interimP);
            } else {
                container.appendChild(p);
            }
        }
        
        p.textContent = text;
        
        // 내용이 추가되었으므로 다운로드 버튼 활성화
        if (text.trim()) {
            container.closest('.log-container-wrapper').querySelector('.log-header').classList.add('has-content');
        }
        
        // 스크롤
        container.scrollTop = container.scrollHeight;
    }

    function translationHandler(data) {
        const { sentence_id, translations } = data;
        
        translations.forEach((transText, i) => {
          const container = logElements[i + 1];
          if (!container || !transText) return;

          // [수정 권장] ID를 기반으로 찾아보고, 없으면 추가하는 것이 더 안정적입니다.
          let p = container.querySelector(`p[data-id="${sentence_id}"]`);
          if (!p) {
              p = document.createElement('p');
              p.dataset.id = sentence_id;
              container.appendChild(p);
          }
          p.textContent = transText;
            
            const logHeader = container.closest('.log-container-wrapper').querySelector('.log-header');
            logHeader.classList.add('has-content');
            container.scrollTop = container.scrollHeight;
        });
    }

    async function listDevices() {
        try {
            await navigator.mediaDevices.getUserMedia({ audio: true });
            const devices = await navigator.mediaDevices.enumerateDevices();
            deviceSelect.innerHTML = "";
            devices.filter(d => d.kind === "audioinput").forEach(d => {
                const opt = document.createElement("option");
                opt.value = d.deviceId;
                opt.textContent = d.label || `마이크 (${d.deviceId.slice(0, 6)})`;
                deviceSelect.appendChild(opt);
            });
        } catch (e) { console.warn(e); }
    }
    const workletCode = `class DownTo16k extends AudioWorkletProcessor{process(e){let l=e[0]?.[0];if(l){let t=new Int16Array(Math.floor(l.length/3));for(let s=0,a=0;s<t.length;s++,a+=3)t[s]=32767*Math.max(-1,Math.min(1,l[a]));this.port.postMessage(t,[t.buffer])}return!0}}registerProcessor("down-to-16k",DownTo16k);`;
    function sendConfigToServer(persist) {
        if (!ws || ws.readyState !== 1) { alert("서버에 연결되지 않았습니다."); return; }
        try {
            const options = JSON.parse(configTextarea.value);
            ws.send(JSON.stringify({ type: "config", options, persist }));
            logSystemMessage(`Config sent (persist=${persist})`);
        } catch (e) {
            alert("잘못된 JSON 형식입니다: " + e.message);
        }
    }
    function sendTranslationConfig() {
        if (!ws || ws.readyState !== 1) return;
        const engine = translationEngineEl.value;
        const target_langs = translationElements.map(item => item.select.value).filter(lang => lang);
        const options = { translation: (engine && target_langs.length > 0) ? { engine, target_langs } : { engine: "" } };
        ws.send(JSON.stringify({ type: "config", options, persist: false }));
    }

    function sendInitialConfigs() {
        if (!ws || ws.readyState !== 1) return;
        const engine = translationEngineEl.value;
        const target_langs = translationElements.map(item => item.select.value).filter(lang => lang);
        const translationOptions = (engine && target_langs.length > 0) ? { engine, target_langs } : { engine: "" };
        const options = { translation: translationOptions };
        ws.send(JSON.stringify({ type: "config", options, persist: false }));
        logSystemMessage("초기 설정(번역)을 서버로 전송했습니다.");
    }
    
    async function cleanupResources() {
        if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
        if (ws) {
            ws.onopen = ws.onmessage = ws.onerror = ws.onclose = null;
            if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) ws.close();
            ws = null;
        }
        if (ac) {
            if (ac.state !== 'closed') await ac.close().catch(console.warn);
            ac = null;
        }
        if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
        if (mediaRecorder) {
            try { if (mediaRecorder.state !== 'inactive') mediaRecorder.stop(); } catch(e){}
        }
        mediaRecorder, recordedChunks, recordingStarted = null, [], false;
        document.getElementById('bar').style.width = "0%";
    }

    function pickRecordingMime() {
        const candidates = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus', 'audio/ogg'];
        for (const c of candidates) if (MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(c)) return c;
        return '';
    }

    function startClientRecording() {
        if (!recordAudioToggle.checked || recordingStarted || !stream) return;
        try {
            mediaRecorder = new MediaRecorder(stream, { mimeType: pickRecordingMime() });
            recordedChunks = [];
            mediaRecorder.ondataavailable = (e) => { if (e.data && e.data.size > 0) recordedChunks.push(e.data); };
            mediaRecorder.onstart = () => { recordingStarted = true; logSystemMessage("클라이언트 녹음 시작."); };
            mediaRecorder.onerror = (e) => { logSystemMessage("클라이언트 녹음 오류: " + e?.error?.message); };
            mediaRecorder.start(1000); 
        } catch (e) { logSystemMessage("클라이언트 녹음 시작 불가: " + e.message); }
    }
    
    function getClientRecordingBlob() {
        return new Promise((resolve) => {
            if (!mediaRecorder || !recordingStarted) return resolve(null);
            if (mediaRecorder.state === 'recording') {
                mediaRecorder.onstop = () => {
                    const blob = new Blob(recordedChunks, { type: recordedChunks[0]?.type || 'audio/webm' });
                    resolve(blob);
                };
                mediaRecorder.stop();
            } else {
                const blob = new Blob(recordedChunks, { type: recordedChunks[0]?.type || 'audio/webm' });
                resolve(blob);
            }
        });
    }

    async function start() {
        await cleanupResources();
        btnStart.disabled = true;
        btnStop.disabled = false;
        bytesSent = 0; seq = 0; 
		//remainingText = '';
        clearLogs(); 
        clearSystemLog();
        if (controlsCard.classList.contains('overlay-visible')) controlsCard.classList.remove('overlay-visible');
        ws = new WebSocket(wsUrlEl.value);
        ws.binaryType = "arraybuffer";
        
        ws.onopen = () => { logSystemMessage("WebSocket connected."); sendInitialConfigs(); };
        ws.onclose = () => { logSystemMessage("WebSocket disconnected."); btnStop.disabled = true; btnStart.disabled = false; };
        ws.onerror = e => logSystemMessage("WebSocket error.");
        
        ws.onmessage = (e) => {
            try {
                const msg = JSON.parse(e.data);
                if (msg.type === "info" || msg.type === "ack") logSystemMessage(msg.text);
                else if (msg.type === "stt_interim") sttInterimHandler(msg);
                else if (msg.type === "stt_final") sttFinalHandler(msg);
                else if (msg.type === "translation_update") translationHandler(msg);
                else if (msg.type === "config" && msg.data) {
                    configTextarea.value = JSON.stringify(msg.data, null, 2);
                    logSystemMessage("서버 설정을 불러왔습니다.");
                }
            } catch(e) { 
                console.error("Error parsing msg:", e);
                logSystemMessage("서버 메시지 처리 오류.");
            }
        };

        try {
            stream = await navigator.mediaDevices.getUserMedia({ audio: { deviceId: { exact: deviceSelect.value }, sampleRate: 48000, channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true }});
            startClientRecording();
            ac = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 48000 });
            await ac.audioWorklet.addModule(URL.createObjectURL(new Blob([workletCode], { type: 'application/javascript' })));
            src = ac.createMediaStreamSource(stream);
            worklet = new AudioWorkletNode(ac, 'down-to-16k');
            worklet.port.onmessage = (e) => {
                if (!ws || ws.readyState !== 1) return;
                const pcm = new Int16Array(e.data);
                const header = new ArrayBuffer(4); new DataView(header).setUint32(0, seq++);
                const payload = new Uint8Array(header.byteLength + pcm.byteLength);
                payload.set(new Uint8Array(header), 0);
                payload.set(new Uint8Array(pcm.buffer), header.byteLength);
                ws.send(payload);
                bytesSent += payload.byteLength;
            };
            src.connect(worklet);
            const analyser = ac.createAnalyser();
            analyser.fftSize = 1024;
            src.connect(analyser);
            const buf = new Float32Array(analyser.fftSize);
            const loop = () => {
                if (!ac || ac.state === 'closed') return;
                analyser.getFloatTimeDomainData(buf);
                const rms = Math.sqrt(buf.reduce((s, v) => s + v * v, 0) / buf.length);
                document.getElementById('bar').style.width = Math.min(100, Math.max(0, (20 * Math.log10(rms) + 60) * (100 / 60))) + "%";
                document.getElementById('stat').textContent = `Sent: ${(bytesSent / 1024).toFixed(1)} KB`;
                rafId = requestAnimationFrame(loop);
            };
            loop();
            logSystemMessage("음성 입력을 시작했습니다.");
        } catch (err) { 
            logSystemMessage("Error starting audio: " + err.message);
            await stop();
        }
    }

    async function stop() {
        btnStop.disabled = true;
        logSystemMessage("중지 요청 중...");
        
        //if (recordAudioToggle.checked && recordingStarted) {
            logSystemMessage("파일 다운로드 준비 중...");
            await downloadAllFilesAsZip();
        //}

        await cleanupResources();
        
        btnStart.disabled = false;
        logSystemMessage("중지되었습니다.");
    }

    document.addEventListener('DOMContentLoaded', () => {
        // --- [수정 시작] id와 user 파라미터를 모두 읽어서 wsUrl을 동적으로 생성 ---
        try {
            const params = new URLSearchParams(window.location.search);
            const wsParams = new URLSearchParams(); // 웹소켓 연결용 파라미터 객체 생성
            
            if (params.has('id')) {
                wsParams.set('id', params.get('id'));
            }
            if (params.has('user')) {
                wsParams.set('user', params.get('user'));
            }

            const wsQueryString = wsParams.toString();
            if (wsQueryString) {
                wsUrlEl.value = `${wsUrlEl.value}?${wsQueryString}`;
            }
        } catch (e) { console.error("URL 파라미터 처리 중 오류:", e); }
        // --- [수정 종료] ---

        listDevices(); 
        
        // --- [수정 시작] 페이지 로드 시 및 이벤트 핸들러 등록 ---
        updateLanguageOptions(); // 페이지 로드 시 초기 언어 목록 설정
        updateTranslationTitles();
        updateWindowsAndLayout();
        
        btnStart.onclick = start; 
        btnStop.onclick = stop;
        sidebarToggleBtn.onclick = toggleSidebar;
        toggleJsonConfigBtn.onclick = () => jsonConfigArea.classList.toggle('visible');		
        
        document.querySelectorAll('.download-btn').forEach(b => b.addEventListener('click', handleDownload));
        toggleCheckboxes.forEach(cb => { cb.onchange = updateWindowsAndLayout; });
        
        // --- [수정] 이벤트 핸들러 로직 개선 ---
        [translationEngineEl, ...translationElements.map(item => item.select)].forEach(el => {
            el.onchange = () => {
                // 1. 엔진이 변경되면, 먼저 언어 옵션 목록부터 갱신
                if (el.id === 'translationEngine') {
                    updateLanguageOptions();
                }

                // 2. 어떤 변경이든(엔진 또는 언어), 선택된 언어를 앞으로 정렬
                compactLanguageSelections();

                // 3. 최종 확정된 설정으로 후속 작업 수행
                sendTranslationConfig();
                updateTranslationTitles();
                updateWindowsAndLayout();
            };
        });
        // --- [수정 종료] ---
        
        window.addEventListener('resize', () => {
            updateOutputGridLayout();

            const isSidebarCollapsed = mainContainer.classList.contains('sidebar-collapsed');
            
            if (window.innerWidth <= 480) {
                if (!isSidebarCollapsed) {
                    toggleSidebar();
                }
            }
        });

        window.dispatchEvent(new Event('resize'));
        
        // --- [추가] QR 코드 모달 로직 (index_vad_trans2_30_bro.php 참고) ---
        const showQrBtn = document.getElementById('showQrBtn');
        const qrModal = document.getElementById('qrModal');
        const closeQrModal = document.getElementById('closeQrModal');
        const qrCodeContainer = document.getElementById('qrCodeContainer');
        let qrCodeInstance = null; 

        showQrBtn.onclick = () => {
            const currentUrl = new URL(window.location.href);
            const userId = currentUrl.searchParams.get('id');
            const viewerPageName = 'index_viewer_google_2.html'; // 공통 뷰어 페이지
            const viewerUrl = `${currentUrl.origin}${currentUrl.pathname.substring(0, currentUrl.pathname.lastIndexOf('/'))}/${viewerPageName}${userId ? '?id=' + encodeURIComponent(userId) : ''}`;
            
            if (qrCodeContainer) {
                qrCodeContainer.innerHTML = '';
                if (qrCodeInstance) {
                    qrCodeInstance.clear();
                }
                qrCodeInstance = new QRCode(qrCodeContainer, {
                    text: viewerUrl,
                    width: 384,
                    height: 384,
                    colorDark : "#000000",
                    colorLight : "#ffffff",
                    correctLevel : QRCode.CorrectLevel.H
                });
            }

            if (qrModal) {
                qrModal.style.display = 'flex';
            }
        };

        closeQrModal.onclick = () => {
            if (qrModal) {
                qrModal.style.display = 'none';
            }
        };

        window.onclick = (event) => {
            if (event.target == qrModal) {
                qrModal.style.display = 'none';
            }
        };
    });
	
	/**
	 * B와 겹치는 A의 "앞부분"을 우선 제거.
	 * - A·B를 토큰화(문장부호 안전) → 공통 접두어 길이를 먼저 계산
	 * - 접두어가 minTokens 이상이면 그만큼만 제거
	 * - 접두어가 없으면, LCSubstring(연속 일치)로 폴백해 가장 긴 구간 한 번 제거
	 */
	function removeFrontOverlap(A, B, opts = {}) {
	  const { caseSensitive = false, minTokens = 1 } = opts;

	  const normalize = (s) =>
		s
		  .replace(/([\p{P}\p{S}])/gu, " $1 ")
		  .replace(/\s+/g, " ")
		  .trim();

	  const joinAndFix = (arr) =>
		arr
		  .join(" ")
		  .replace(/\s+([\p{P}\p{S}])/gu, "$1")
		  .replace(/([.!?…])(\p{L})/gu, "$1 $2")
		  .trim();

	  let sA = normalize(A);
	  let sB = normalize(B);
	  if (!caseSensitive) {
		sA = sA.toLowerCase();
		sB = sB.toLowerCase();
	  }
	  const tokensA = sA ? sA.split(" ") : [];
	  const tokensB = sB ? sB.split(" ") : [];
	  const n = tokensA.length, m = tokensB.length;

	  if (!n || !m) return { result: A, removed: "", startA: -1, endA: -1 };

	  // 1) 공통 "접두어" 길이 계산
	  let prefixLen = 0;
	  while (prefixLen < n && prefixLen < m && tokensA[prefixLen] === tokensB[prefixLen]) {
		prefixLen++;
	  }

	  // 1-1) 접두어가 충분하면 그만큼만 제거
	  if (prefixLen >= minTokens) {
		const removedTokens = tokensA.slice(0, prefixLen);
		const remainTokens = tokensA.slice(prefixLen);
		return {
		  result: joinAndFix(remainTokens.length ? remainTokens : []),
		  removed: joinAndFix(removedTokens),
		  startA: 0,
		  endA: prefixLen - 1,
		};
	  }

	  // 2) 접두어가 없을 때만 폴백: LCSubstring(토큰 기준 최장 연속 일치) 한 구간 제거
	  const dp = Array.from({ length: n + 1 }, () => Array(m + 1).fill(0));
	  let maxLen = 0, endIndexA = -1;
	  for (let i = 1; i <= n; i++) {
		for (let j = 1; j <= m; j++) {
		  if (tokensA[i - 1] === tokensB[j - 1]) {
			dp[i][j] = dp[i - 1][j - 1] + 1;
			if (dp[i][j] > maxLen) {
			  maxLen = dp[i][j];
			  endIndexA = i - 1;
			}
		  }
		}
	  }
	  if (maxLen < minTokens || endIndexA === -1) {
		return { result: A, removed: "", startA: -1, endA: -1 };
	  }
	  const startIndexA = endIndexA - maxLen + 1;
	  const removedTokens = tokensA.slice(startIndexA, endIndexA + 1);
	  const remainTokens = [...tokensA.slice(0, startIndexA), ...tokensA.slice(endIndexA + 1)];

	  return {
		result: joinAndFix(remainTokens),
		removed: joinAndFix(removedTokens),
		startA: startIndexA,
		endA: endIndexA,
	  };
	}


	
</script>
</body>
</html>