import { useEffect, useRef, useState } from 'react'
import { api } from './api.js'
import Conversations from './Conversations.jsx'
import Message from './Message.jsx'
import Sidebar from './Sidebar.jsx'

const SUGGESTIONS = {
  ask: [
    '¿Cuántos días de vacaciones tengo al año?',
    '¿Quién aprueba un pedido de 10.000 euros?',
    '¿Cuándo es obligatorio el arnés anticaídas?',
    '¿Cuál es la capital de Francia?'
  ],
  agent: [
    '¿Qué documentos hay?',
    'Resume el manual de seguridad',
    'Compara la política de vacaciones con el procedimiento de compras'
  ]
}

export default function App() {
  const [health, setHealth] = useState(null)
  const [documents, setDocuments] = useState([])
  const [selected, setSelected] = useState([])
  const [messages, setMessages] = useState([])
  const [mode, setMode] = useState('ask')
  const [question, setQuestion] = useState('')
  const [loading, setLoading] = useState(false)
  const [toast, setToast] = useState(null)
  const [conversations, setConversations] = useState([])
  const [conversationId, setConversationId] = useState(null)
  const bottom = useRef(null)

  async function refresh() {
    const [h, docs, convs] = await Promise.all([api.health(), api.documents(), api.conversations()])
    setHealth(h)
    setDocuments(docs)
    setConversations(convs)
    setSelected(current => current.filter(id => docs.some(d => d.id === id)))
  }

  const refreshConversations = () => api.conversations().then(setConversations).catch(() => {})

  async function openConversation(id) {
    try {
      const detail = await api.conversation(id)
      setConversationId(id)
      setMessages(
        detail.messages.map(m =>
          m.role === 'user'
            ? { role: 'user', text: m.content }
            : {
                role: 'assistant',
                data: {
                  answer: m.content,
                  sources: m.sources,
                  steps: m.steps,
                  grounded: m.grounded,
                  search_query: m.search_query
                }
              }
        )
      )
    } catch (error) {
      setToast(error.message)
    }
  }

  function newConversation() {
    setConversationId(null)
    setMessages([])
  }

  async function deleteConversation(conv) {
    if (!window.confirm(`¿Eliminar la conversación "${conv.title}"?`)) return
    try {
      await api.removeConversation(conv.id)
      if (conv.id === conversationId) newConversation()
      await refreshConversations()
    } catch (error) {
      setToast(error.message)
    }
  }

  useEffect(() => {
    refresh().catch(() => setToast('No se puede conectar con la API. ¿Está arrancado el backend?'))
  }, [])

  useEffect(() => {
    if (messages.length) bottom.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages, loading])

  useEffect(() => {
    if (!toast) return
    const timer = setTimeout(() => setToast(null), 5000)
    return () => clearTimeout(timer)
  }, [toast])

  async function send(text) {
    const q = text.trim()
    if (!q || loading) return
    setQuestion('')
    setMessages(m => [...m, { role: 'user', text: q }])
    setLoading(true)
    try {
      const data =
        mode === 'agent' ? await api.agent(q, conversationId) : await api.ask(q, selected, conversationId)
      setMessages(m => [...m, { role: 'assistant', data }])
      setConversationId(data.conversation_id)
      refreshConversations()
    } catch (error) {
      setMessages(m => [...m, { role: 'assistant', error: error.message }])
    } finally {
      setLoading(false)
    }
  }

  const toggle = id => setSelected(s => (s.includes(id) ? s.filter(x => x !== id) : [...s, id]))
  const demo = health?.mode === 'demo'

  return (
    <div className="layout">
      <header className="topbar">
        <div className="brand">
          <span className="logo" aria-hidden="true">C</span>
          <div>
            <h1>Citadoc</h1>
            <span className="tagline">Pregunta a tus documentos · RAG con citas</span>
          </div>
        </div>
        <div className="topbar-right">
          <a className="api-link" href="/docs" target="_blank" rel="noreferrer">API · Swagger</a>
          {health && (
            <span className={`badge ${demo ? 'badge-demo' : 'badge-live'}`} title={`Embeddings: ${health.embeddings} · Índice: ${health.vector_store}`}>
              {demo ? 'Modo demo · sin claves de API' : `LLM: ${health.mode.replace('llm:', '')}`}
            </span>
          )}
        </div>
      </header>

      <aside className="sidebar">
        <Conversations
          items={conversations}
          activeId={conversationId}
          onSelect={openConversation}
          onNew={newConversation}
          onDelete={deleteConversation}
        />
        <Sidebar
          documents={documents}
          selected={selected}
          onToggle={toggle}
          onChange={refresh}
          onError={setToast}
        />
      </aside>

      <main className="chat">
        <div className="mode-switch" role="tablist" aria-label="Modo">
          <button role="tab" aria-selected={mode === 'ask'} className={mode === 'ask' ? 'is-on' : ''} onClick={() => setMode('ask')}>
            Preguntar
          </button>
          <button role="tab" aria-selected={mode === 'agent'} className={mode === 'agent' ? 'is-on' : ''} onClick={() => setMode('agent')}>
            Agente con herramientas
          </button>
        </div>

        <div className="messages">
          {messages.length === 0 && (
            <div className="empty">
              <h2>{mode === 'agent' ? 'El agente elige la herramienta' : 'Respuestas con su fuente'}</h2>
              <p>
                {mode === 'agent'
                  ? 'Puede listar, buscar, resumir o comparar documentos. Verás cada paso que da.'
                  : 'Cada afirmación lleva un número que abre el fragmento original. Si la respuesta no está en los documentos, lo dice en lugar de inventar.'}
              </p>
              {demo && (
                <p className="demo-note">
                  En modo demo no hay un LLM: la respuesta se compone con las frases más relevantes de los documentos. Con una clave de API, un modelo redacta la respuesta usando los mismos fragmentos.
                </p>
              )}
              <div className="suggestions">
                {SUGGESTIONS[mode].map(s => (
                  <button key={s} onClick={() => send(s)}>{s}</button>
                ))}
              </div>
            </div>
          )}
          {messages.map((m, i) => <Message key={i} message={m} />)}
          {loading && <div className="message assistant loading">Buscando en los documentos…</div>}
          <div ref={bottom} />
        </div>

        <form className="composer" onSubmit={e => { e.preventDefault(); send(question) }}>
          <input
            value={question}
            onChange={e => setQuestion(e.target.value)}
            placeholder={mode === 'agent' ? 'Pide algo: resume, compara, busca…' : 'Escribe tu pregunta…'}
            aria-label="Pregunta"
            maxLength={2000}
          />
          <button type="submit" disabled={loading || !question.trim()}>Enviar</button>
        </form>
      </main>

      {toast && <div className="toast" role="alert">{toast}</div>}
    </div>
  )
}
