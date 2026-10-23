import { useState } from 'react'

const TOOL_LABELS = {
  list_documents: 'Listar documentos',
  search_documents: 'Buscar en documentos',
  summarize_document: 'Resumir documento',
  compare_documents: 'Comparar documentos'
}

/** Turns [n] markers into buttons that open source n. */
function withCitations(text, onCite, active) {
  return text.split(/(\[\d+\])/g).map((part, j) => {
    const match = part.match(/^\[(\d+)\]$/)
    if (!match) return part
    const ref = Number(match[1])
    return (
      <button
        key={j}
        className={`cite ${active === ref ? 'is-active' : ''}`}
        onClick={() => onCite(ref)}
        aria-label={`Ver fuente ${ref}`}
      >
        {ref}
      </button>
    )
  })
}

/** Plain text with "- " bullet lines rendered as a list. */
function AnswerText({ text, onCite, active }) {
  const blocks = []
  for (const line of text.split('\n')) {
    const bullet = line.match(/^\s*[-•*]\s+(.*)$/)
    const last = blocks[blocks.length - 1]
    if (bullet) {
      if (last?.type === 'list') last.items.push(bullet[1])
      else blocks.push({ type: 'list', items: [bullet[1]] })
    } else if (line.trim()) {
      blocks.push({ type: 'p', text: line })
    }
  }
  return blocks.map((block, i) =>
    block.type === 'list' ? (
      <ul key={i}>{block.items.map((item, j) => <li key={j}>{withCitations(item, onCite, active)}</li>)}</ul>
    ) : (
      <p key={i}>{withCitations(block.text, onCite, active)}</p>
    )
  )
}

export default function Message({ message }) {
  const [active, setActive] = useState(null)
  if (message.role === 'user') {
    return <div className="message user">{message.text}</div>
  }
  if (message.error) {
    return <div className="message assistant error">{message.error}</div>
  }

  const { answer, sources = [], steps, grounded, search_query: searchQuery } = message.data
  const toggle = ref => setActive(current => (current === ref ? null : ref))

  return (
    <div className={`message assistant ${grounded === false ? 'not-found' : ''}`}>
      {steps?.length > 0 && (
        <ol className="steps">
          {steps.map((step, i) => (
            <li key={i}>
              <span className="step-tool">{TOOL_LABELS[step.tool] || step.tool}</span>
              {Object.values(step.input).length > 0 && (
                <span className="step-input">{Object.values(step.input).join(' · ')}</span>
              )}
            </li>
          ))}
        </ol>
      )}

      {searchQuery && (
        <p className="search-query" title="Pregunta de seguimiento completada con la conversación">
          Buscado como: «{searchQuery}»
        </p>
      )}

      <div className="answer">
        <AnswerText text={answer} onCite={toggle} active={active} />
      </div>

      {sources.length > 0 && (
        <div className="sources">
          <span className="sources-title">Fuentes</span>
          {sources.map(source => (
            <div key={source.chunk_id} className={`source ${active === source.ref ? 'is-open' : ''}`}>
              <button className="source-head" onClick={() => toggle(source.ref)}>
                <span className="cite">{source.ref}</span>
                <span className="source-name">{source.doc_name}</span>
                {source.page && <span className="source-page">pág. {source.page}</span>}
              </button>
              {active === source.ref && <blockquote>{source.snippet}</blockquote>}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
