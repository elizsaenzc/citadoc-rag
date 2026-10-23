const formatDate = iso => {
  const date = new Date(iso)
  const today = new Date()
  return date.toDateString() === today.toDateString()
    ? date.toLocaleTimeString('es-ES', { hour: '2-digit', minute: '2-digit' })
    : date.toLocaleDateString('es-ES', { day: 'numeric', month: 'short' })
}

export default function Conversations({ items, activeId, onSelect, onNew, onDelete }) {
  return (
    <section className="conversations">
      <div className="section-head">
        <h2>Conversaciones</h2>
        <button className="new-chat" onClick={onNew}>+ Nueva</button>
      </div>
      {items.length === 0 ? (
        <p className="hint">Tus preguntas se guardan aquí para retomarlas.</p>
      ) : (
        <ul className="conv-list">
          {items.map(conv => (
            <li key={conv.id} className={conv.id === activeId ? 'is-active' : ''}>
              <button className="conv-open" onClick={() => onSelect(conv.id)} title={conv.title}>
                <span className="conv-title">{conv.title}</span>
                <span className="conv-meta">{formatDate(conv.updated_at)} · {conv.messages / 2} preg.</span>
              </button>
              <button className="icon-button" onClick={() => onDelete(conv)} aria-label={`Eliminar ${conv.title}`}>
                ×
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}
