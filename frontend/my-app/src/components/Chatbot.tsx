'use client';

import { useEffect, useRef, useState } from 'react';

interface Message {
  sender: 'user' | 'bot';
  text: string;
  timestamp: string;
}

export default function Chatbot() {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const ws = useRef<WebSocket | null>(null);
  const botMessageIndex = useRef<number | null>(null);

  useEffect(() => {
    if (isOpen && !ws.current) {
      ws.current = new WebSocket('ws://localhost:8000/ws/chat');

      ws.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.answer && botMessageIndex.current !== null) {
          setMessages((prev) => {
            const updated = [...prev];
            const i = botMessageIndex.current!;
            updated[i] = {
              ...updated[i],
              text: updated[i].text + data.answer,
            };
            return updated;
          });
          setIsTyping(false);
        }
      };
    }

    return () => {
      ws.current?.close();
      ws.current = null;
    };
  }, [isOpen]);

  const sendMessage = (msg: string) => {
    if (ws.current && msg.trim()) {
      const timestamp = new Date().toLocaleTimeString();
      const userMessage: Message = { sender: 'user', text: msg, timestamp };
      const botPlaceholder: Message = { sender: 'bot', text: '', timestamp };

      setMessages((prev) => {
        const updated = [...prev, userMessage, botPlaceholder];
        botMessageIndex.current = updated.length - 1;
        return updated;
      });

      setInput('');
      setIsTyping(true);
      ws.current.send(JSON.stringify({ session_id: 'session-1', message: msg }));
    }
  };

  return (
    <div className="fixed bottom-6 right-6 z-50">
      {isOpen ? (
        <div className="w-80 bg-white border rounded shadow-lg flex flex-col overflow-hidden">
          <div className="bg-blue-700 text-white px-4 py-2 flex justify-between items-center">
            <span className="font-semibold">Chatbot</span>
            <button onClick={() => setIsOpen(false)} className="text-white font-bold text-lg">×</button>
          </div>
          <div className="p-4 text-sm">
            <p className="font-semibold mb-2">Hi, how can I help you?</p>
            <div className="max-h-40 overflow-y-auto space-y-2 mb-2">
              {messages.map((msg, idx) => (
                <div key={idx} className={`text-${msg.sender === 'user' ? 'right' : 'left'}`}>
                  <span
                    className={`inline-block px-3 py-2 rounded-lg ${
                      msg.sender === 'user'
                        ? 'bg-blue-500 text-white'
                        : 'bg-gray-200 text-black'
                    }`}
                  >
                    {msg.text}
                  </span>
                  <span className="text-xs text-gray-500 ml-2">{msg.timestamp}</span>
                </div>
              ))}
            </div>
            {isTyping && <div className="text-gray-500">Typing...</div>}
            <div className="flex border-t pt-2">
              <input
                type="text"
                placeholder="Ask our chatbot"
                className="flex-1 p-2 border rounded-l text-sm"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && sendMessage(input)}
              />
              <button
                onClick={() => sendMessage(input)}
                className="bg-blue-600 text-white px-4 py-2 rounded-r text-sm"
              >
                ➤
              </button>
            </div>
          </div>
        </div>
      ) : (
        <button
          onClick={() => setIsOpen(true)}
          className="bg-blue-700 text-white px-4 py-2 rounded shadow-lg"
        >
          Chat
        </button>
      )}
    </div>
  );
}
