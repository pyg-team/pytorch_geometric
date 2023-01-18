function warnOnLatestVersion() {
  if (!window.READTHEDOCS_DATA || window.READTHEDOCS_DATA.version !== "latest") {
    return;  // not on ReadTheDocs and not latest.
  }

  var note = document.createElement('div');
  note.setAttribute('class', 'admonition note');
  note.innerHTML = "<p class='first admonition-title'>Note</p> " +
    "<p> " +
    "This documentation is for an <b>unreleased development version</b>. " +
    "Click <a href='/en/stable'><b>here</b></a> to access the documentation of the current stable release." +
    "</p>";

  var parent = document.querySelector('#pyg-documentation');
  if (parent)
    parent.insertBefore(note, parent.querySelector('h1'));
}

document.addEventListener('DOMContentLoaded', warnOnLatestVersion);
