let isLoading = false;

function toggleSettings() {
    const panel = document.getElementById('settingsPanel');
    panel.classList.toggle('active');
}

function updateValue(id) {
    const element = document.getElementById(id);
    const valueSpan = document.getElementById(id + 'Value');
    valueSpan.textContent = element.value;
}

function handleKeyPress(event) {
    if (event.key === 'Enter' && !isLoading) {
        sendMessage();
    }
}

function addMessage(content, isUser) {
    const messagesContainer = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = `<strong>${isUser ? 'Vous' : 'Bot'}:</strong> ${content}`;
    
    messageDiv.appendChild(contentDiv);
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function addLoadingMessage() {
    const messagesContainer = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot-message';
    messageDiv.id = 'loadingMessage';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = '<strong>Bot:</strong> <span class="loading">En train de réfléchir</span>';
    
    messageDiv.appendChild(contentDiv);
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function removeLoadingMessage() {
    const loadingMsg = document.getElementById('loadingMessage');
    if (loadingMsg) {
        loadingMsg.remove();
    }
}

async function sendMessage() {
    const input = document.getElementById('userInput');
    const message = input.value.trim();
    
    if (!message || isLoading) {
        return;
    }
    
    addMessage(message, true);
    input.value = '';
    
    isLoading = true;
    const sendBtn = document.getElementById('sendBtn');
    sendBtn.disabled = true;
    sendBtn.textContent = 'Envoi...';
    
    addLoadingMessage();
    
    const maxTokens = parseInt(document.getElementById('maxTokens').value);
    const temperature = parseFloat(document.getElementById('temperature').value);
    const topK = parseInt(document.getElementById('topK').value);
    const topP = parseFloat(document.getElementById('topP').value);
    const repetitionPenalty = parseFloat(document.getElementById('repetitionPenalty').value);
    
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                max_tokens: maxTokens,
                temperature: temperature,
                top_k: topK,
                top_p: topP,
                repetition_penalty: repetitionPenalty
            })
        });
        
        const data = await response.json();
        
        removeLoadingMessage();
        
        if (data.success) {
            addMessage(data.response, false);
        } else {
            addMessage(`Erreur: ${data.error}`, false);
        }
    } catch (error) {
        removeLoadingMessage();
        addMessage(`Erreur de connexion: ${error.message}`, false);
    } finally {
        isLoading = false;
        sendBtn.disabled = false;
        sendBtn.textContent = 'Envoyer';
        input.focus();
    }
}

async function loadModelConfig() {
    try {
        const response = await fetch('/api/config');
        const config = await response.json();
        
        const modelInfo = document.getElementById('modelInfo');
        modelInfo.innerHTML = `
            <h4>Configuration du modèle</h4>
            <p>
                <strong>Taille vocabulaire:</strong> ${config.vocab_size}<br>
                <strong>Dimension embeddings:</strong> ${config.embed_dim}<br>
                <strong>Têtes d'attention:</strong> ${config.num_heads}<br>
                <strong>Nombre de couches:</strong> ${config.num_layers}<br>
                <strong>Longueur séquence max:</strong> ${config.max_seq_len}
            </p>
        `;
    } catch (error) {
        console.error('Erreur lors du chargement de la config:', error);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    loadModelConfig();
    document.getElementById('userInput').focus();
});
