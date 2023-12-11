function handleLabelClick(labelId, inputId) {
    const label = document.getElementById(labelId);
    const input = document.getElementById(inputId);

    label.style.display = 'none';
    input.style.visibility = 'visible';
    input.focus();
}

function handleInput(labelId, inputId) {
    const label = document.getElementById(labelId);
    const input = document.getElementById(inputId);

    label.style.display = input.value ? 'none' : 'inline-block';
}