<script lang="ts">
  import type { AttachmentRef } from "./api";

  interface Props {
    attachments: AttachmentRef[];
    /** Where the session's attachment is served, by sha256. */
    url: (sha: string) => string;
    /** Why the row carries them, for an image's alt text. */
    origin: string;
    testid: string;
  }

  let { attachments, url, origin, testid }: Props = $props();
</script>

<div class="attachments" data-testid={testid}>
  {#each attachments as attachment (attachment.sha256)}<a
      href={url(attachment.sha256)}
      target="_blank"
      rel="noopener"
      title={attachment.media_type}
      >{#if attachment.media_type.startsWith("image/")}<img
          src={url(attachment.sha256)}
          alt={origin}
          loading="lazy"
        />{:else}<span class="document" data-testid="attachment-document">PDF</span>{/if}</a
    >{/each}
</div>

<style>
  .attachments {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5em;
    margin-top: 0.5em;
  }
  img {
    max-width: 16em;
    max-height: 12em;
    border: 1px solid var(--muted);
  }
  .document {
    display: inline-block;
    padding: 0.6em 0.9em;
    border: 1px solid var(--muted);
    border-radius: 0.3rem;
    font-family: ui-monospace, monospace;
  }
</style>
