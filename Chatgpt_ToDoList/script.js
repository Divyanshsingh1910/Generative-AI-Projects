// Selectors
const toDoForm = document.querySelector('#to-do-form');
const toDoList = document.querySelector('#to-do-list');

// Event Listeners
toDoForm.addEventListener('submit', addToDo);
toDoList.addEventListener('click', deleteToDo);

// Functions
function addToDo (event) {
  event.preventDefault();
  const toDoInput = toDoForm.querySelector('input[type="text"]');
  const newToDo = document.createElement('li');
  const toDoCheck = document.createElement('input');
  toDoCheck.type = 'checkbox';
  const toDoLabel = document.createElement('label');
  toDoLabel.innerText = toDoInput.value;
  const toDoButton = document.createElement('button');
  toDoButton.innerText = 'Delete';
  toDoButton.className = 'delete-btn';
  newToDo.appendChild(toDoCheck);
  newToDo.appendChild(toDoLabel);
  newToDo.appendChild(toDoButton);
  toDoList.appendChild(newToDo);
  toDoInput.value = '';
}

function deleteToDo (event) {
  const item = event.target;
  if (item.classList[0] === 'delete-btn') {
    const toDo = item.parentElement;
    toDo.remove();
  }
  if (item.type === 'checkbox') {
    const toDoLabel = item.nextElementSibling;
    if (item.checked) {
      toDoLabel.style.textDecoration = 'line-through';
    } else {
      toDoLabel.style.textDecoration = 'none';
    }
  }
}
