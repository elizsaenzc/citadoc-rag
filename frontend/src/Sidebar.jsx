import { useRef, useState } from 'react'
import { api } from './api.js'

const ACCEPTED = '.pdf,.docx,.txt,.md'

export default function Sidebar({ documents, selected, onToggle, onChange, onError }) {
  const input = useRef(null)
  const [uploading, setUploading] = useState(false)
  const [dragging, setDragging] = useState(false)

  async function upload(files) {
    if (!files?.length) return
    setUploading(true)
    try {
      for (const file of files) await api.upload(file)
      await onChange()
    } catch (error) {
      onError(error.message)
    } finally {
      setUploading(false)
      if (input.current) input.current.value = ''
    }
  }

  async function remove(doc) {
    if (!window.confirm(`¿Eliminar "${doc.name}" del índice?`)) return
    try {
      await api.remove(doc.id)
      await onChange()
    } catch (error) {
      onError(error.message)
    }
  }

  return (
    <section className="documents">
      <h2>Documentos</h2>
      <label
        className={`dropzone ${dragging ? 'is-dragging' : ''}`}
        onDragOver={e => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={e => { e.preventDefault(); setDragging(false); upload(e.dataTransfer.files) }}
      >
        <input ref={input} type="file" accept={ACCEPTED} multiple onChange={e => upload(e.target.files)} />
        <strong>{uploading ? 'Indexando…' : 'Sube un documento'}</strong>
        <span>PDF, DOCX, TXT o MD · arrastra o haz clic</span>
      </label>

      <p className="hint">
        {selected.length
          ? `Buscando solo en ${selected.length} documento${selected.length > 1 ? 's' : ''}`
          : 'Buscando en todos los documentos. Marca alguno para filtrar.'}
      </p>

      <ul className="doc-list">
        {documents.map(doc => (
          <li key={doc.id} className={selected.includes(doc.id) ? 'is-selected' : ''}>
            <label>
              <input type="checkbox" checked={selected.includes(doc.id)} onChange={() => onToggle(doc.id)} />
              <span className="doc-name" title={doc.name}>{doc.name}</span>
            </label>
            <span className="doc-meta">
              {doc.chunks} fragm.{doc.pages ? ` · ${doc.pages} pág.` : ''}{doc.sample ? ' · ejemplo' : ''}
            </span>
            <button className="icon-button" onClick={() => remove(doc)} aria-label={`Eliminar ${doc.name}`}>
              ×
            </button>
          </li>
        ))}
      </ul>
    </section>
  )
}
