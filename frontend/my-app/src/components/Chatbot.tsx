"use client";

import { useState } from "react";

export default function Chatbot() {
  const [messages, setMessages] = useState<{ user: string; bot: string }[]>([]);
  const [input, setInput] = useState("");

  const sessionId = "session-001";

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = input.trim();
    setMessages([...messages, { user: userMessage, bot: "..." }]);
    setInput("");

    try {
      const res = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          message: userMessage,
        }),
      });

      const data = await res.json();
      const newBotMessage = data.answer;

      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1].bot = newBotMessage;
        return updated;
      });
    } catch (err) {
      console.error("Error:", err);
    }
  };

  return (
    <div style={{ maxWidth: "600px", margin: "2rem auto", padding: "1rem" }}>
      <h1 className="text-xl font-bold mb-4">Retail Chatbot</h1>
      <div className="min-h-[300px] border rounded-md p-4 mb-4 bg-white shadow">
        {messages.map((msg, i) => (
          <div key={i} className="mb-3">
            <p><strong>You:</strong> {msg.user}</p>
            <p><strong>Bot:</strong> {msg.bot}</p>
          </div>
        ))}
      </div>
      <div className="flex gap-2">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
          className="flex-1 border px-3 py-2 rounded shadow"
          placeholder="Ask our chatbot"
        />
        <button
          onClick={sendMessage}
          className="bg-blue-600 text-white px-4 py-2 rounded shadow hover:bg-blue-700"
        >
          Send
        </button>
      </div>
    </div>
  );
}
