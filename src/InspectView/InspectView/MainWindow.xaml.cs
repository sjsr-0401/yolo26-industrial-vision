using InspectView.ViewModels;
using System.Linq;
using System.Windows;

namespace InspectView;

public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();
        Closed += (_, _) => (DataContext as MainViewModel)?.Dispose();
    }

    private async void Window_Drop(object sender, DragEventArgs e)
    {
        if (e.Data.GetDataPresent(DataFormats.FileDrop))
        {
            var files = (string[])e.Data.GetData(DataFormats.FileDrop)!;
            var images = files.Where(f =>
                f.EndsWith(".jpg", System.StringComparison.OrdinalIgnoreCase) ||
                f.EndsWith(".jpeg", System.StringComparison.OrdinalIgnoreCase) ||
                f.EndsWith(".png", System.StringComparison.OrdinalIgnoreCase) ||
                f.EndsWith(".bmp", System.StringComparison.OrdinalIgnoreCase))
                .ToArray();

            if (images.Length > 0 && DataContext is MainViewModel vm)
            {
                await vm.ProcessImages(images);
            }
        }
    }

    private void Window_DragOver(object sender, DragEventArgs e)
    {
        e.Effects = e.Data.GetDataPresent(DataFormats.FileDrop)
            ? DragDropEffects.Copy
            : DragDropEffects.None;
        e.Handled = true;
    }
}
