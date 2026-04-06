/**
 * DiseasePredict — Chatbot Frontend
 */
document.addEventListener('DOMContentLoaded', () => {
    const chatHtml = `
        <div id="chatbot-container" class="chatbot-collapsed">
            <div id="chatbot-header">
                <span>Health Assistant</span>
                <button id="chatbot-toggle">−</button>
            </div>
            <div id="chatbot-messages"></div>
            <div id="chatbot-input-area">
                <input type="text" id="chatbot-input" placeholder="Ask a question...">
                <button id="chatbot-send">Send</button>
            </div>
        </div>
    `;
    document.body.insertAdjacentHTML('beforeend', chatHtml);

    const container = document.getElementById('chatbot-container');
    const messages = document.getElementById('chatbot-messages');
    const input = document.getElementById('chatbot-input');
    const sendBtn = document.getElementById('chatbot-send');
    const toggleBtn = document.getElementById('chatbot-toggle');

    const addMessage = (text, isUser) => {
        const msg = document.createElement('div');
        msg.className = `chat-msg ${isUser ? 'user-msg' : 'bot-msg'}`;
        msg.textContent = text;
        messages.appendChild(msg);
        messages.scrollTop = messages.scrollHeight;
    };

    const sendMessage = async () => {
        const text = input.value.trim();
        if (!text) return;
        addMessage(text, true);
        input.value = '';

        const res = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text })
        });
        const data = await res.json();
        addMessage(data.response, false);
    };

    sendBtn.onclick = sendMessage;
    input.onkeypress = (e) => e.key === 'Enter' && sendMessage();
    toggleBtn.onclick = () => container.classList.toggle('chatbot-collapsed');
});